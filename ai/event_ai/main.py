import sys
import os
import time
from datetime import datetime
import requests

# [1] 경로 설정: 최상위 ai/ 폴더를 기준으로 모듈을 찾도록 설정
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

try:
    from event_ai.inference import run_ai_inference, init_inference_engine
    from event_ai.packet_parser import PacketParser
    from event_ai.buffer import AirQualityBuffer
    # 절대 경로 임포트
    from activity_ai.smart_activity import SmartActivityDetector
except ImportError as e:
    print(f" 모듈 임포트 실패: {e}")
    sys.exit(1)

def main():
    # --- 1. 초기화 ---
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # [A] 공기질 센서 파서 (시리얼 연결)
    try:
        parser = PacketParser(port='/dev/serial0', baudrate=115200)
        print("센서 연결 성공: /dev/serial0")
    except Exception as e:
        print(f"센서 연결 실패: {e}")
        parser = None

    # [B] 센서 AI 엔진 (모델 & 스케일러 로드)
    model_path = os.path.join(base_path, "event_model.pt")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    engine = init_inference_engine(model_path=model_path, scaler_path=scaler_path)

    # [C] YOLO 비전 모듈 (객체 생성 필수)
    vision_module = SmartActivityDetector()
    
    # [D] 데이터 버퍼 (30분 롤링)
    aq_buffer = AirQualityBuffer(max_len=900)

    # 관리용 변수
    air_packet_count = 0
    last_display = time.time()
    last_inference = time.time()

    # 설정값 (Thresholds)
    # PM2.5 기울기 \Delta > 0.5 일 때 AI 추론 시작
    PM25_SLOPE_THRESHOLD = 0.5  
    PROB_THRESHOLD = 0.70       
    CLOUD_URL = "https://your-cloud-server.com/api/v1/upload"

    print("\n" + "="*50)
    print("ARIA AI 시스템 가동: [센서 + 비전 통합 모드]")
    print("="*50)

    try:
        while True:
            # [2] 센서 데이터 수집
            if parser:
                packet = parser.read_packet()
                if packet and packet.get('type') == 'AIR':
                    air_packet_count += 1
                    aq_buffer.add_data(
                        temp=packet['temp'],
                        humi=packet['humi'],
                        pm25=packet['pm25'],
                        voc=packet['voc']
                    )

            now = time.time()

            # [3] AI 추론 및 비전 검증 (2초 주기로 체크)
            if now - last_inference >= 2.0:
                # 버퍼가 어느 정도 쌓였을 때만 실행 (최소 5분 데이터)
                if len(aq_buffer.buffer) >= 150:
                    features = aq_buffer.get_session_features()

                    # PM2.5 급증 감지 시 추론 실행
                    if features and features['pm25_slope'] > PM25_SLOPE_THRESHOLD:
                        prob = run_ai_inference(engine, aq_buffer.buffer)

                        # 요리 확률이 임계치를 넘으면 비전 검증 트리거
                        if prob is not None and prob >= PROB_THRESHOLD:
                            print(f"\n[AI 경보] 요리 확률 {prob*100:.1f}%! 비전 검증을 시작합니다.")

                            # [C4-1] YOLO 비전 검증 (객체를 통해 메서드 호출)
                            yolo_res = vision_module.detect_cooking_event(
                                corridor_video="data/videos/current_corridor.mp4",
                                kitchen_video="data/videos/current_kitchen.mp4"
                            )

                            print(f"YOLO 결과: {yolo_res['reason']} (확정: {yolo_res['confirmed']})")

                            # [C4-2] 데이터 원샷 패키징
                            snapshot = aq_buffer.get_full_logs()
                            final_package = {
                                "meta": {
                                    "timestamp": datetime.now().isoformat(),
                                    "predicted_prob": float(prob),
                                    "yolo_verified": yolo_res['confirmed'],
                                    "stats": features
                                },
                                "raw_logs": snapshot
                            }

                            # [C4-3] 클라우드 전송
                            try:
                                # response = requests.post(CLOUD_URL, json=final_package, timeout=10)
                                print(f"[Cloud] {len(snapshot)}개의 데이터 패키지를 전송했습니다.")
                            except Exception as e:
                                print(f"전송 실패: {e}")

                last_inference = now

            # [4] 상태 요약 리포트 (5초 주기)
            if now - last_display >= 5.0:
                current_pm = aq_buffer.buffer[-1]['pm25'] if aq_buffer.buffer else 'N/A'
                print(f"[Status] Buffer: {len(aq_buffer.buffer)}/900 | "
                      f"Air Packets: {air_packet_count} | "
                      f"Current PM2.5: {current_pm}")
                last_display = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n시스템을 안전하게 종료합니다.")
    finally:
        if parser and parser.ser:
            parser.ser.close()

if __name__ == "__main__":
    main()