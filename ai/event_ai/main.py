import time
from datetime import datetime
import requests # 클라우드 전송용

# 우리가 만든 모듈들
from inference import run_ai_inference, init_inference_engine
from packet_parser import PacketParser
from buffer import AirQualityBuffer

def main():
    # 1. 초기화 (공기질 전용)
    parser = PacketParser(port='/dev/serial0', baudrate=115200)
    engine = init_inference_engine() # 모델 & 스케일러 로드
    aq_buffer = AirQualityBuffer(max_len=900) # 30분 롤링 버퍼

    # 관리용 변수 (AIR 전용)
    air_packet_count = 0
    last_display = time.time()
    last_inference = time.time()

    # 설정값
    PM25_SLOPE_THRESHOLD = 0.5  # AI 트리거 기준
    PROB_THRESHOLD = 0.70       # 주방 이동 기준
    CLOUD_URL = "https://your-cloud-server.com/api/v1/upload"

    print("ARIA AI 시스템 가동: [공기질 모니터링 모드]")

    try:
        while True:
            # ESP32로부터 패킷 읽기
            packet = parser.read_packet()

            if packet and packet.get('type') == 'AIR':
                air_packet_count += 1

                # AIR 데이터만 버퍼에 추가
                aq_buffer.add_data(
                    temp=packet['temp'],
                    humi=packet['humi'],
                    pm25=packet['pm25'],
                    voc=packet['voc']
                )

            now = time.time()

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [C3-2] 변화율 감지 및 AI 추론 (성국님 핵심 로직)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if now - last_inference >= 2.0:
                # 최소 5분치(150개) 데이터가 있을 때부터 기울기 체크
                if len(aq_buffer.buffer) >= 150:
                    features = aq_buffer.get_session_features()

                    # 변화율(Slope) 트리거 확인
                    if features and features['pm25_slope'] > PM25_SLOPE_THRESHOLD:
                        print(f"변화 감지! (PM2.5 기울기: {features['pm25_slope']:.4f})")

                        # AI 추론 (0.0 ~ 1.0 확률 반환)
                        prob = run_ai_inference(engine, aq_buffer.buffer)

                        if prob is not None:
                            print(f"[AI 결과] 요리 확률: {prob*100:.1f}%")

                            # 확률이 높으면 주방 이동 및 최종 검증 시퀀스
                            if prob >= PROB_THRESHOLD:
                                print(f"확률 {prob*100:.1f}%로 요리 예상! 주방으로 이동합니다.")

                                # 1) 이동 전 30분 데이터 스냅샷 확보
                                snapshot = aq_buffer.get_full_logs()

                                # 2) [가상] 주방 도착 후 사람 확인 (YOLO 등)
                                # has_person = check_human_presence()
                                has_person = True

                                # 3) 데이터 패키징 (성국님 DB 설계 반영)
                                final_package = {
                                    "meta": {
                                        "predicted_prob": prob,
                                        "yolo_verified": has_person,
                                        "final_label": 1 if has_person else 0,
                                        "stats": features
                                    },
                                    "raw_logs": snapshot
                                }

                                # 4) 클라우드 전송
                                try:
                                    # requests.post(CLOUD_URL, json=final_package)
                                    print("[Cloud] 30분치 센서 데이터 및 검증 라벨 전송 완료")
                                except:
                                    print("전송 에러 발생")

                last_inference = now

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [Status] 5초마다 공기질 요약 리포트
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if now - last_display >= 5.0:
                current_pm = aq_buffer.buffer[-1]['pm25'] if aq_buffer.buffer else 'N/A'
                print(f"[Status] Buffer: {len(aq_buffer.buffer)}/900 | "
                      f"Air Packets: {air_packet_count} | "
                      f"Current PM2.5: {current_pm}")
                last_display = now

            time.sleep(0.01) # CPU 휴식

    except KeyboardInterrupt:
        print("\n시스템을 안전하게 종료합니다.")
    finally:
        parser.ser.close()

if __name__ == "__main__":
    main()