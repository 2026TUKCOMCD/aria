import sys
import os
import time
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

current_file_path = os.path.abspath(__file__)
# core -> event_ai -> ai -> project_root 순으로 네 단계 위로 이동
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "ai"))

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

    # [E] 데이터 관리 모듈 (YOLO 결과 JSON 저장 + 영상 자동 삭제)
    dm = DataManager()

    # [F] 저전력 모드 관리 (23:00~07:00 수면 시간에 AI 분석 스킵)
    power_manager = PowerManager(sleep_time="23:00", wake_time="07:00", verbose=False)

    # [G] 데이터 버퍼 (30분 롤링)
    aq_buffer = AirQualityBuffer(max_len=900)

    # [H] 로컬 백업 폴더
    backup_dir = os.path.join(base_path, "..", "failed_logs")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 이동/녹화 시간 상수
    # TODO(ROS2 노드 변환 시): NAV_TO_KITCHEN_SEC 제거 → Nav2 goal 완료 신호로 대체
    NAV_TO_KITCHEN_SEC = 10    # 임시: 주방까지 직진으로 대체, 실제는 Nav2 이동 시간
    KITCHEN_RECORD_SEC = 29.0  # 주방 360도 스캔 시간

    # 관리 변수
    air_packet_count = 0
    last_display    = time.time()
    last_inference  = time.time()
    is_analyzing    = False
    last_event_time = 0

    # --- [Threshold Settings] ---
    PM25_SLOPE_THRESHOLD = 0.5
    VOC_SLOPE_THRESHOLD  = 0.3
    PM25_HIGH_THRESHOLD  = 100.0
    VOC_HIGH_THRESHOLD   = 150.0
    PROB_THRESHOLD   = 0.70
    COOLDOWN_SECONDS = 60

    print("\n" + "="*50)
    print(f"ARIA AI 시스템 가동: [ID: {ROBOT_ID}]")
    print("보안 업링크 및 데이터 아카이빙 활성화")
    print("="*50)

    try:
        while True:
            # [1] 센서 데이터 수집
            if parser:
                packet = parser.read_packet()
                if packet and packet.get('type') == 'AIR':
                    air_packet_count += 1
                    aq_buffer.add_data(
                        temp=packet['temp'], humi=packet['humi'],
                        pm25=packet['pm25'], voc=packet['voc']
                    )

            now = time.time()

            # [2] AI 추론 및 비전 검증 (2초 주기)
            if now - last_inference >= 2.0:
                last_inference = now

                if len(aq_buffer.buffer) >= 150:
                    # 저전력 모드 체크: 수면 시간(23:00~07:00)에는 분석 스킵
                    if not power_manager.should_process_activity():
                        pass
                    else:
                        features = aq_buffer.get_session_features()

                        if features:
                            is_pm_slope = features.get('pm25_slope', 0) > PM25_SLOPE_THRESHOLD
                            is_voc_slope = features.get('voc_slope', 0) > VOC_SLOPE_THRESHOLD
                            is_pm_high   = features.get('current_pm25', 0) > PM25_HIGH_THRESHOLD
                            is_voc_high  = features.get('current_voc', 0) > VOC_HIGH_THRESHOLD

                            if (is_pm_slope or is_voc_slope or is_pm_high or is_voc_high) and not is_analyzing:
                                if now - last_event_time >= COOLDOWN_SECONDS:
                                    is_analyzing = True

                                    reasons = []
                                    if is_pm_slope:  reasons.append("PM기울기")
                                    if is_voc_slope: reasons.append("VOC기울기")
                                    if is_pm_high:   reasons.append("PM고농도")
                                    if is_voc_high:  reasons.append("VOC고농도")

                                    print(f"\n[트리거 감지: {' & '.join(reasons)}] 정밀 분석 시작...")

                                    # 2단계: AI 추론
                                    prob_res = run_ai_inference(engine, aq_buffer.buffer)
                                    prob = prob_res.get('cooking', 0.0)
                                    print(f"[AI 분석 결과] 요리 확률: {prob*100:.1f}%")

                                    if prob >= PROB_THRESHOLD:
                                        print("-> 요리 확률 임계치 초과. 영상 녹화 시작...")

                                        # ── [1단계] 주방으로 출발 + 동시에 경로 녹화 ──
                                        # 출발과 동시에 녹화 시작, 도착하면 녹화 종료
                                        # TODO(ROS2 노드): send_forward → Nav2 goal로 교체
                                        #   개선 목표: 목적지 2m 전부터 녹화 시작
                                        print("[이동] 주방으로 출발 + 경로 녹화 시작")
                                        recorder.start_recording(mode="corridor")
                                        if parser:
                                            parser.send_forward(speed=70, duration_sec=NAV_TO_KITCHEN_SEC)
                                        else:
                                            time.sleep(NAV_TO_KITCHEN_SEC)
                                        corridor_path = recorder.stop_recording()
                                        print(f"[Camera] 경로 영상 저장 완료: {corridor_path}")
                                        time.sleep(1.0)

                                        # ── [2단계] 주방 도착 후 360도 스캔 + 녹화 ──
                                        print(f"[Camera] 주방 360도 영상 녹화 중... ({KITCHEN_RECORD_SEC}초)")
                                        recorder.start_recording(mode="360")
                                        if parser:
                                            parser.send_rotate_360(speed=50, clockwise=True, duration_sec=KITCHEN_RECORD_SEC)
                                        else:
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

                                        # ── 데이터 패키징 ────────────────────────────
                                        final_package = aq_buffer.make_package(
                                            ROBOT_ID, prob, yolo_res['confirmed'], features
                                        )

                                        # ── 클라우드 전송 ────────────────────────────
                                        success = False
                                        for attempt in range(3):
                                            try:
                                                headers = {
                                                    'Content-Type': 'application/json',
                                                    'X-ARIA-SECRET': SECRET_TOKEN
                                                }
                                                response = requests.post(
                                                    CLOUD_URL, json=final_package,
                                                    headers=headers, timeout=15
                                                )
                                                if response.status_code == 200:
                                                    res_data = response.json()
                                                    print(f"[Cloud] 전송 성공! (S3 경로: {res_data.get('path', 'N/A')})")
                                                    success = True
                                                    last_event_time = time.time()
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

                                        # ── 영상 메타데이터 저장 + 영상 자동 삭제 ──────
                                        dm.save_cooking_event(
                                            result=yolo_res,
                                            corridor_video=corridor_path,
                                            kitchen_video=kitchen_path
                                        )

                                    is_analyzing = False

            # [3] 상태 요약 (5초 주기)
            if now - last_display >= 5.0:
                if aq_buffer.buffer:
                    last_data = aq_buffer.buffer[-1]
                    temp = last_data.get('temperature', 'N/A')
                    humi = last_data.get('humidity', 'N/A')
                    pm25 = last_data.get('pm25', 'N/A')
                    voc  = last_data.get('voc', 'N/A')
                    mode_status = power_manager.get_status()['mode']
                    print(f"\n[Status] {datetime.now().strftime('%H:%M:%S')} | {mode_status}")
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
        recorder.cleanup()
        if parser and parser.ser:
            parser.ser.close()


if __name__ == "__main__":
    main()
