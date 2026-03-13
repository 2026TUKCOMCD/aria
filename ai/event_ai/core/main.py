#!/usr/bin/env python3
"""
ARIA AI 메인 시스템

ESP32 공기질 센서 데이터 기반 요리 이벤트 감지 및 YOLO 검증

Author: 박진주
"""

import sys
import os
import time
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

# 경로 설정
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(project_root)

try:
    from ai.event_ai.core.inference import run_ai_inference, init_inference_engine
    from ai.event_ai.core.packet_parser import PacketParser
    from ai.event_ai.core.buffer import AirQualityBuffer
    from activity_ai.smart_activity import SmartActivityDetector
    from activity_ai.camera_recorder import CameraRecorder
    from activity_ai.data_manager import DataManager
    from activity_ai.power_manager import PowerManager
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    print(f"현재 탐색 경로(sys.path): {sys.path}")
    sys.exit(1)

# 환경 변수 로드
dotenv_path = os.path.join(project_root, "ai", "event_ai", ".env")
load_dotenv(dotenv_path)


def main():
    # --- 1. 초기화 ---
    base_path = os.path.dirname(os.path.abspath(__file__))

    CLOUD_URL = os.getenv("ARIA_LOG_API_URL")
    SECRET_TOKEN = os.getenv("ARIA_SECRET_TOKEN")
    ROBOT_ID = os.getenv("ROBOT_ID", "1")

    if not CLOUD_URL or not SECRET_TOKEN:
        print("에러: 환경 변수(URL 또는 Token)가 설정되지 않았습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    # [A] 공기질 센서 파서
    try:
        parser = PacketParser(port='/dev/serial0', baudrate=115200)
        print("센서 연결 성공: /dev/serial0")
    except Exception as e:
        print(f"센서 연결 실패: {e}")
        parser = None

    # [B] 센서 AI 엔진
    model_path = os.path.join(base_path, "..", "models", "event_model.pt")
    scaler_path = os.path.join(base_path, "..", "models", "scaler.pkl")
    engine = init_inference_engine(model_path=model_path, scaler_path=scaler_path)

    # [C] YOLO 비전 모듈
    vision_module = SmartActivityDetector()

    # [D] 카메라 녹화 모듈 (1640x1232는 메모리 부족 → 640x480 고정)
    recorder = CameraRecorder(output_dir="/home/jj/aria/data/videos", resolution=(640, 480))

    # [E] 데이터 관리 모듈
    dm = DataManager()

    # [F] 저전력 모드 관리
    power_manager = PowerManager(sleep_time="23:00", wake_time="07:00")

    # [G] 데이터 버퍼 (30분 롤링)
    aq_buffer = AirQualityBuffer(max_len=900)

    # [H] 로컬 백업 폴더
    backup_dir = os.path.join(base_path, "..", "failed_logs")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 관리 변수
    air_packet_count = 0
    last_display = time.time()
    last_inference = time.time()

    # 임계치 설정
    PM25_SLOPE_THRESHOLD = 0.5
    PROB_THRESHOLD = 0.70

    # 녹화 시간 설정 (초)
    CORRIDOR_RECORD_SEC = 10
    KITCHEN_RECORD_SEC  = 10

    print("\n" + "="*50)
    print(f"ARIA AI 시스템 가동: [ID: {ROBOT_ID}]")
    print("="*50)

    try:
        while True:
            # [2] 센서 데이터 수집
            if parser:
                packet = parser.read_packet()
                if packet and packet.get('type') == 'AIR':
                    air_packet_count += 1
                    aq_buffer.add_data(
                        temp=packet['temp'], humi=packet['humi'],
                        pm25=packet['pm25'], voc=packet['voc']
                    )

            now = time.time()

            # [3] AI 추론 (2초 주기)
            if now - last_inference >= 2.0:
                last_inference = now

                if len(aq_buffer.buffer) >= 150:
                    features = aq_buffer.get_session_features()

                    # 1단계: PM2.5 기울기 급증 확인
                    if features and features['pm25_slope'] > PM25_SLOPE_THRESHOLD:

                        # 2단계: 저전력 모드 체크
                        if not power_manager.should_process_activity():
                            print("[PowerManager] SLEEP_MODE - 요리 감지 스킵")
                            continue

                        # 3단계: AI 추론
                        prob_res = run_ai_inference(engine, aq_buffer.buffer)
                        prob = prob_res.get('cooking', 0.0)

                        # 4단계: 요리 확률 임계치 초과 시 영상 녹화 + YOLO 트리거
                        if prob >= PROB_THRESHOLD:
                            print(f"\n[AI 경보] 요리 확률 {prob*100:.1f}%! 영상 녹화 시작...")

                            # ── 경로 영상 녹화 (10초) ──────────────────────────
                            print(f"[Camera] 경로 영상 녹화 중... ({CORRIDOR_RECORD_SEC}초)")
                            recorder.start_recording(mode="corridor")
                            time.sleep(CORRIDOR_RECORD_SEC)
                            corridor_path = recorder.stop_recording()
                            print(f"[Camera] 경로 영상 저장 완료: {corridor_path}")

                            # ── 주방 360도 영상 녹화 (회전하면서 동시 녹화, 10초) ──
                            print(f"[Camera] 주방 360도 영상 녹화 + 회전 시작... ({KITCHEN_RECORD_SEC}초)")
                            recorder.start_recording(mode="360")
                            if parser:
                                parser.send_rotate_360(
                                    speed=150,
                                    clockwise=True,
                                    duration_sec=KITCHEN_RECORD_SEC
                                )
                            else:
                                # parser 없으면 그냥 대기
                                time.sleep(KITCHEN_RECORD_SEC)
                            kitchen_path = recorder.stop_recording()
                            print(f"[Camera] 주방 영상 저장 완료: {kitchen_path}")

                            # 5단계: YOLO 검증
                            yolo_res = vision_module.detect_cooking_event(
                                corridor_video=corridor_path,
                                kitchen_video=kitchen_path
                            )
                            print(f"YOLO 결과: {yolo_res['reason']} (확정: {yolo_res['confirmed']})")

                            # 6단계: 데이터 패키징
                            final_package = aq_buffer.make_package(
                                robot_id=ROBOT_ID,
                                predicted_prob=prob,
                                yolo_verified=yolo_res['confirmed'],
                                features=features
                            )

                            # 7단계: 클라우드 전송
                            success = False
                            max_retries = 3

                            for attempt in range(max_retries):
                                try:
                                    headers = {
                                        'Content-Type': 'application/json',
                                        'X-ARIA-SECRET': SECRET_TOKEN
                                    }
                                    response = requests.post(
                                        CLOUD_URL,
                                        json=final_package,
                                        headers=headers,
                                        timeout=15
                                    )

                                    if response.status_code == 200:
                                        res_data = response.json()
                                        print(f"[Cloud] 전송 성공! (S3 경로: {res_data.get('path', 'N/A')})")
                                        success = True
                                        break
                                    else:
                                        print(f"[Cloud] 전송 실패 (시도 {attempt+1}/{max_retries}): {response.status_code}")
                                except Exception as e:
                                    print(f"[Cloud] 네트워크 오류: {e}")

                                if attempt < max_retries - 1:
                                    time.sleep(2)

                            # 최종 실패 시 로컬 백업
                            if not success:
                                backup_filename = f"fail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                backup_path = os.path.join(backup_dir, backup_filename)
                                with open(backup_path, 'w', encoding='utf-8') as f:
                                    json.dump(final_package, f, ensure_ascii=False, indent=4)
                                print(f"데이터 로컬 백업 완료: {backup_path}")

                            # 8단계: DataManager로 결과 저장 및 영상 삭제
                            dm.save_cooking_event(
                                result=yolo_res,
                                corridor_video=corridor_path,
                                kitchen_video=kitchen_path
                            )

            # [4] 상태 요약 (5초 주기)
            if now - last_display >= 5.0:
                if aq_buffer.buffer:
                    last_data = aq_buffer.buffer[-1]
                    temp = last_data.get('temperature') or last_data.get('temp', 'N/A')
                    humi = last_data.get('humidity') or last_data.get('humi', 'N/A')
                    pm25 = last_data.get('pm25', 'N/A')
                    voc  = last_data.get('voc', 'N/A')
                    status = power_manager.get_status()
                    print(f"\n[Status] {datetime.now().strftime('%H:%M:%S')}")
                    print(f" 온도: {temp}°C | 습도: {humi}%")
                    print(f" PM2.5: {pm25} µg/m³ | VOC: {voc} ppm")
                    print(f" Buffer: {len(aq_buffer.buffer)}/900 | Packet: {air_packet_count} | Mode: {status['mode']}")
                    print("-" * 45)
                else:
                    print("[Status] 센서 데이터를 기다리는 중...")
                last_display = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n시스템을 종료합니다.")
    finally:
        recorder.cleanup()
        if parser and parser.ser:
            parser.ser.close()


if __name__ == "__main__":
    main()