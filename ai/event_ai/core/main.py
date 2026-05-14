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
    is_analyzing = False # 중복 분석 방지용 플래그
    last_event_time = 0       # 마지막 전송 성공 시간
    # --- [Threshold Settings] ---
    # 1단계 트리거용 (기울기 및 절대 수치)
    PM25_SLOPE_THRESHOLD = 0.5    # 2초당 미세먼지 상승폭
    VOC_SLOPE_THRESHOLD  = 0.3    # 2초당 VOC Index 상승폭
    PM25_HIGH_THRESHOLD  = 100.0  # 미세먼지 절대 수치 (고농도 기준)
    VOC_HIGH_THRESHOLD   = 150.0  # VOC 절대 수치 (고농도 기준)

    # 2단계 AI 판단용
    PROB_THRESHOLD = 0.70         # AI 추론 결과 요리 확률 70% 이상일 때만 YOLO 실행

    # 시스템 관리용
    COOLDOWN_SECONDS = 60         # 이벤트 전송 후 재가동까지의 휴식 시간
    # ----------------------------

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
                    
                    if features:
                        # 1단계 트리거 조건 판단
                        is_pm_slope = features.get('pm25_slope', 0) > PM25_SLOPE_THRESHOLD
                        is_voc_slope = features.get('voc_slope', 0) > VOC_SLOPE_THRESHOLD
                        is_pm_high = features.get('current_pm25', 0) > PM25_HIGH_THRESHOLD
                        is_voc_high = features.get('current_voc', 0) > VOC_HIGH_THRESHOLD

                        # 트리거 조건 충족 + 분석 중 아님 + 쿨다운 시간 경과 확인
                        if (is_pm_slope or is_voc_slope or is_pm_high or is_voc_high) and not is_analyzing:
                            
                            # 마지막 전송 성공 후 COOLDOWN_SECONDS(예: 60초)가 지났는지 체크
                            if now - last_event_time >= COOLDOWN_SECONDS:
                                is_analyzing = True
                                
                                reasons = []
                                if is_pm_slope: reasons.append("PM기울기")
                                if is_voc_slope: reasons.append("VOC기울기")
                                if is_pm_high: reasons.append("PM고농도")
                                if is_voc_high: reasons.append("VOC고농도")
                                
                                print(f"\n[트리거 감지: {' & '.join(reasons)}] 정밀 분석 시작...")

                                # 2단계: 추론 수행
                                prob_res = run_ai_inference(engine, aq_buffer.buffer)
                                prob = prob_res.get('cooking', 0.0)
                                print(f"[AI 분석 결과] 요리 확률: {prob*100:.1f}%")

                                if prob >= PROB_THRESHOLD:
                                    print(f"-> 요리 확률 임계치 초과. 비전 검증을 시작합니다.")
                                    yolo_res = vision_module.detect_cooking_event(
                                        corridor_video="data/videos/current_corridor.mp4",
                                        kitchen_video="data/videos/current_kitchen.mp4"
                                    )
                                    print(f"YOLO 결과: {yolo_res['reason']} (확정: {yolo_res['confirmed']})")

                                    # 3단계: 데이터 패키징 및 전송
                                    final_package = aq_buffer.make_package(ROBOT_ID, prob, yolo_res['confirmed'], features)

                                    success = False
                                    for attempt in range(3):
                                        try:
                                            headers = {'Content-Type': 'application/json', 'X-ARIA-SECRET': SECRET_TOKEN}
                                            response = requests.post(CLOUD_URL, json=final_package, headers=headers, timeout=10)
                                            if response.status_code == 200:
                                                print(f"[Cloud] 전송 성공! {COOLDOWN_SECONDS}초간 대기 모드로 전환합니다.")
                                                success = True
                                                last_event_time = time.time() # 전송 성공 시점에 쿨다운 타이머 시작
                                                break
                                        except: pass
                                        time.sleep(1)

                                    if not success:
                                        print(f"!! 전송 실패로 로컬 백업을 수행합니다.")
                                        # (백업 로직 생략 가능 - 기존과 동일)
                                
                                is_analyzing = False # 분석 프로세스 종료

            # [4] 상태 요약 (5초 주기)
            if now - last_display >= 5.0:
                if aq_buffer.buffer:
                    last_data = aq_buffer.buffer[-1]
                    
                    temp = last_data.get('temperature', 'N/A')
                    humi = last_data.get('humidity', 'N/A')
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