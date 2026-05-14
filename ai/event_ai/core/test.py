#!/usr/bin/env python3
"""
ARIA AI 테스트용 메인 - 모든 트리거 강제 True
(요리 이벤트 전체 플로우 확인용)

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
    sys.exit(1)

# 환경 변수 로드
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


def main():
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

    # [D] 카메라 녹화 모듈
    recorder = CameraRecorder(output_dir="/home/jj/aria/data/videos", resolution=(640, 480))

    # [E] 데이터 관리 모듈
    dm = DataManager()

    # [F] 저전력 모드 관리
    power_manager = PowerManager(sleep_time="23:00", wake_time="07:00")

    # [G] 데이터 버퍼
    aq_buffer = AirQualityBuffer(max_len=900)

    # [H] 로컬 백업 폴더
    backup_dir = os.path.join(base_path, "..", "failed_logs")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    CORRIDOR_RECORD_SEC = 10
    KITCHEN_RECORD_SEC  = 22.0

    print("\n" + "="*50)
    print(f"ARIA AI 테스트 모드 가동: [ID: {ROBOT_ID}]")
    print("⚠️  모든 트리거 강제 True - 요리 이벤트 즉시 실행")
    print("="*50)

    # 버퍼에 더미 데이터 채우기 (150개 필요)
    print("[TEST] 버퍼 더미 데이터 채우는 중...")
    for _ in range(150):
        aq_buffer.add_data(temp=25.0, humi=50.0, pm25=80.0, voc=200.0)
    print(f"[TEST] 버퍼 준비 완료: {len(aq_buffer.buffer)}개")

    try:
        # ── 트리거 1: buffer >= 150 → True (이미 채움)
        # ── 트리거 2: pm25_slope > 0.5 → True (강제)
        # ── 트리거 3: should_process_activity() → True (강제)
        # ── 트리거 4: prob >= 0.70 → True (강제 1.0)

        features = aq_buffer.get_session_features()
        if features is None:
            features = {}
        features['pm25_slope'] = 999.0  # 강제 급증

        prob = 1.0  # 요리 확률 100% 강제

        print(f"\n[AI 경보] 요리 확률 {prob*100:.1f}%! 영상 녹화 시작...")

        # ── 경로 영상 녹화 ──────────────────────────
        print(f"[Camera] 경로 영상 녹화 중... ({CORRIDOR_RECORD_SEC}초)")
        recorder.start_recording(mode="corridor")
        time.sleep(CORRIDOR_RECORD_SEC)
        corridor_path = recorder.stop_recording()
        print(f"[Camera] 경로 영상 저장 완료: {corridor_path}")

        # ── 주방 360도 영상 녹화 + 모터 회전 ──────────
        print(f"[Camera] 주방 360도 영상 녹화 + 회전 시작... ({KITCHEN_RECORD_SEC}초)")
        recorder.start_recording(mode="360")
        if parser:
            parser.send_rotate_360(speed=50, clockwise=True, duration_sec=KITCHEN_RECORD_SEC)
        else:
            print("[TEST] parser 없음 → 회전 스킵, 대기만")
            time.sleep(KITCHEN_RECORD_SEC)
        kitchen_path = recorder.stop_recording()
        print(f"[Camera] 주방 영상 저장 완료: {kitchen_path}")

        # ── YOLO 검증 ────────────────────────────────
        print("[YOLO] 영상 분석 중...")
        yolo_res = vision_module.detect_cooking_event(
            corridor_video=corridor_path,
            kitchen_video=kitchen_path
        )
        print(f"[YOLO] 결과: {yolo_res['reason']} (확정: {yolo_res['confirmed']})")

        # ── 데이터 패키징 ─────────────────────────────
        final_package = aq_buffer.make_package(
            robot_id=ROBOT_ID,
            predicted_prob=prob,
            yolo_verified=yolo_res['confirmed'],
            features=features
        )

        # ── 클라우드 전송 ─────────────────────────────
        print("[Cloud] 클라우드 전송 중...")
        success = False
        for attempt in range(3):
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'X-ARIA-SECRET': SECRET_TOKEN
                }
                response = requests.post(CLOUD_URL, json=final_package, headers=headers, timeout=15)
                if response.status_code == 200:
                    res_data = response.json()
                    print(f"[Cloud] 전송 성공! (S3 경로: {res_data.get('path', 'N/A')})")
                    success = True
                    break
                else:
                    print(f"[Cloud] 전송 실패 (시도 {attempt+1}/3): {response.status_code}")
            except Exception as e:
                print(f"[Cloud] 네트워크 오류: {e}")
            if attempt < 2:
                time.sleep(2)

        if not success:
            backup_filename = f"fail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(backup_dir, backup_filename)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(final_package, f, ensure_ascii=False, indent=4)
            print(f"[Cloud] 전송 실패 → 로컬 백업: {backup_path}")

        # ── DataManager 저장 및 영상 삭제 ────────────
        #dm.save_cooking_event(
         #   result=yolo_res,
         #   corridor_video=corridor_path,
         #   kitchen_video=kitchen_path
        #)

        print("\n" + "="*50)
        print("✅ 테스트 완료! 전체 플로우 정상 동작 확인")
        print("="*50)

    except KeyboardInterrupt:
        print("\n테스트를 중단합니다.")
    finally:
        recorder.cleanup()
        if parser and parser.ser:
            parser.ser.close()


if __name__ == "__main__":
    main()