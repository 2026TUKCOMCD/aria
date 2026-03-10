import sys
import os
import time
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
# [1] 경로 설정 최적화
# 현재 파일 위치: event_ai/core/main.py
# project_root는 ai/ 폴더가 있는 최상위 경로를 가리키게 설정합니다.
current_file_path = os.path.abspath(__file__)
# core -> event_ai -> ai(root) 순으로 세 단계 위로 이동
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

try:
    # 패키지 구조에 맞춘 절대 임포트
    from ai.event_ai.core.inference import run_ai_inference, init_inference_engine
    from ai.event_ai.core.packet_parser import PacketParser
    from ai.event_ai.core.buffer import AirQualityBuffer
    from activity_ai.smart_activity import SmartActivityDetector
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    print(f"현재 탐색 경로(sys.path): {sys.path}")
    sys.exit(1)

# 환경 변수 로드 (.env 위치 대응)
# event_ai 폴더 또는 프로젝트 루트에 있는 .env를 탐색합니다.
dotenv_path = os.path.join(project_root, "ai", "event_ai", ".env")
load_dotenv(dotenv_path)
def main():
    # --- 1. 초기화 및 설정 ---
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 환경 변수에서 클라우드 설정 로드 (보안 강화)
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

    # [D] 데이터 버퍼 (30분 롤링)
    aq_buffer = AirQualityBuffer(max_len=900)

    # [E] 로컬 백업 폴더
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

    print("\n" + "="*50)
    print(f"ARIA AI 시스템 가동: [ID: {ROBOT_ID}]")
    print("보안 업링크 및 데이터 아카이빙 활성화")
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

            # [3] AI 추론 및 비전 검증 (2초 주기)
            if now - last_inference >= 2.0:
                last_inference = now

                if len(aq_buffer.buffer) >= 150:
                    features = aq_buffer.get_session_features()

                    # 1단계: 기울기 급증 시 AI 가동
                    if features and features['pm25_slope'] > PM25_SLOPE_THRESHOLD:
                        prob_res = run_ai_inference(engine, aq_buffer.buffer)
                        prob = prob_res.get('cooking', 0.0)

                        # 2단계: AI 확률 임계치 초과 시 비전(YOLO) 트리거
                        if prob >= PROB_THRESHOLD:
                            print(f"\n[AI 경보] 요리 확률 {prob*100:.1f}%! 비전 검증 시작...")

                            yolo_res = vision_module.detect_cooking_event(
                                corridor_video="data/videos/current_corridor.mp4",
                                kitchen_video="data/videos/current_kitchen.mp4"
                            )
                            print(f"YOLO 결과: {yolo_res['reason']} (확정: {yolo_res['confirmed']})")

                            # 3단계: 데이터 원샷 패키징 (C4-2: buffer.py의 메서드 활용)
                            final_package = aq_buffer.make_package(
                                robot_id=ROBOT_ID,
                                predicted_prob=prob,
                                yolo_verified=yolo_res['confirmed'],
                                features=features
                            )

                            # 4단계: 클라우드 전송 및 안정성 검증 (C4-3)
                            success = False
                            max_retries = 3

                            for attempt in range(max_retries):
                                try:
                                    headers = {
                                        'Content-Type': 'application/json',
                                        'X-ARIA-SECRET': SECRET_TOKEN # 보안 헤더 추가
                                    }
                                    response = requests.post(CLOUD_URL, json=final_package, headers=headers, timeout=15)

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

            # [4] 상태 요약 (5초 주기) - 모든 센서 데이터 출력 버전
            # [4] 상태 요약 (5초 주기)
            if now - last_display >= 5.0:
                if aq_buffer.buffer:
                    last_data = aq_buffer.buffer[-1]
                    
                    temp = last_data.get('temperature') or last_data.get('temp', 'N/A')
                    humi = last_data.get('humidity') or last_data.get('humi', 'N/A')
                    pm25 = last_data.get('pm25', 'N/A')
                    voc = last_data.get('voc', 'N/A')
                    
                    print(f"\n[Status] {datetime.now().strftime('%H:%M:%S')}")
                    print(f" 온도: {temp}°C | 습도: {humi}%")
                    print(f" PM2.5: {pm25} µg/m³ | VOC: {voc} ppm")
                    print(f" Buffer: {len(aq_buffer.buffer)}/900 | Packet Count: {air_packet_count}")
                    print("-" * 45)
                else:
                    print("[Status] 센서 데이터를 기다리는 중...")
                
                last_display = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n시스템을 종료합니다.")
    finally:
        if parser and parser.ser:
            parser.ser.close()

if __name__ == "__main__":
    main()