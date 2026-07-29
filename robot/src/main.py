import time
import json
import os
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# 작성해주신 최종본 파서 임포트
from packet_parser import PacketParser 

# ==================================================================
# 1. AWS IoT Core 설정 (인증서 경로는 실제 위치에 맞게 유지)
# ==================================================================
ENDPOINT = "adecukeeb0iln-ats.iot.ap-northeast-2.amazonaws.com"
CLIENT_ID = "1"
PATH_TO_ROOT = "certs/AmazonRootCA1.pem"
PATH_TO_CERT = "certs/bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-certificate.pem.crt"
PATH_TO_KEY  = "certs/bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-private.pem.key"

# ==================================================================
# 2. 클라우드 명령 수신 콜백 (명세서 기준 양방향 제어)
# ==================================================================
def customCallback(client, userdata, message):
    print(f"\n [명령 수신] 토픽: {message.topic}")
    try:
        payload = json.loads(message.payload.decode('utf-8'))
        print(f"   내용: {payload}")
        
        # 클라우드에서 정지 명령이 내려왔을 때 파서의 제어 함수 호출 예시
        if payload.get("action") == "TURN_OFF":
            print(" 시스템: 정지 명령을 수행합니다.")
            parser.send_stop()
            
    except Exception as e:
        print(f"명령 처리 에러: {e}")


# ==================================================================
# 2-2. Shadow 상태 변경 수신 콜백 (웹에서 설정값을 바꿨을 때)
# ==================================================================
def shadowDeltaCallback(client, userdata, message):
    print(f"\n[Shadow 변경 알림 수신] 토픽: {message.topic}")
    try:
        # payload 안에는 웹에서 바꾼 '상태값'만 딱 들어있습니다.
        payload = json.loads(message.payload.decode('utf-8'))
        delta_state = payload.get("state", {})
        print(f"  변경해야 할 값: {delta_state}")

        # 1. 모드(MODE) 변경이 들어왔을 때
        if "mode" in delta_state:
            new_mode = delta_state["mode"]
            current_status["mode"] = new_mode # 라즈베리파이의 현재 상태 바구니 업데이트
            print(f"로봇 모드를 [{new_mode}] 로 변경합니다.")
            
            # (필요하다면) ESP32 파서에게 모드 변경 명령 전송 
            # 예: parser.send_...() 

        # 2. 전원(POWER) 변경이 들어왔을 때
        if "power" in delta_state:
            new_power = delta_state["power"]
            current_status["power"] = new_power
            print(f"전원 상태를 [{new_power}] 로 변경합니다.")
            if new_power == "OFF":
                parser.send_stop()

        # 3. 터보(TURBO), SLAM 등 다른 상태값들도 똑같이 추가 가능...

    except Exception as e:
        print(f"Shadow 델타 처리 에러: {e}")


# ==================================================================
# 3. 메인 시스템 초기화
# ==================================================================
print("ARIA 로봇 메인 시스템 가동 준비 중...")

# 파서 객체 생성 (포트 이름이 /dev/serial0 인지 /dev/ttyUSB0 인지 환경에 맞게 확인)
parser = PacketParser(port='/dev/serial0', baudrate=115200)

myMQTTClient = AWSIoTMQTTClient(CLIENT_ID)
myMQTTClient.configureEndpoint(ENDPOINT, 8883)
myMQTTClient.configureCredentials(PATH_TO_ROOT, PATH_TO_KEY, PATH_TO_CERT)

# 비용 절감 핵심 셋팅 1: Offline Queue 비활성화
# 인터넷이 끊겼을 때 쌓이는 메시지를 0으로 설정하여 무시 (재연결 시 폭탄 과금 방지)
myMQTTClient.configureOfflinePublishQueueing(0)
myMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)

#LWT 설정
lwt_topic = f"aria/{CLIENT_ID}/presence"
lwt_payload = json.dumps({"status": "Offline", "reason": "Connection Lost"})
myMQTTClient.configureLastWill(lwt_topic, lwt_payload, 1)

#Keep Alive Signal 설정
KEEP_ALIVE_SEC = 60
if myMQTTClient.connect(keepAliveIntervalSecs=KEEP_ALIVE_SEC):
    print("AWS IoT Core 연결 완료 (Keep-Alive 및 LWT 활성화)")
else:
    print("연결 실패")
    exit()

# 클라우드 명령(cmd) 수신 대기
subscribe_topic = f"aria/{CLIENT_ID}/cmd/#"
myMQTTClient.subscribe(subscribe_topic, 1, customCallback)

shadow_delta_topic = f"$aws/things/{CLIENT_ID}/shadow/update/delta"
myMQTTClient.subscribe(shadow_delta_topic, 1, shadowDeltaCallback)

# 센서 데이터 전송 토픽
publish_topic = f"aria/{CLIENT_ID}/data/status"

# 클라우드에 전송할 상태를 담아두는 딕셔너리 (명세서 규격)
current_status = {
    "battery": 100,
    "power": "ON",
    "is_charging": False,
    "mode": "AUTO",
    "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
    "sensors": {"pm25": 0, "voc": 0, "temperature": 0.0, "humidity": 0.0}
}

last_publish_time = time.time()

# 비용 절감 핵심 셋팅 2: 전송 주기 5초 (월 260만 건 -> 월 50만 건으로 최적화)
PUBLISH_INTERVAL = 5.0  

# ==================================================================
# 4. 무한 루프 (시리얼 파싱 및 딜레이 전송)
# ==================================================================
try:
    while True:
        # 1. 시리얼 버퍼에서 데이터 읽기 (논블로킹이므로 멈추지 않음)
        packet = parser.read_packet()
        
        # 2. 패킷 종류에 맞춰 current_status 변수만 최신화 (아직 전송 안 함)
        if packet:
            p_type = packet['type']
            if p_type == 'AIR':
                current_status["sensors"]["pm25"] = packet["pm25"]
                current_status["sensors"]["voc"] = packet["voc"]
                current_status["sensors"]["temperature"] = packet["temp"]
                current_status["sensors"]["humidity"] = packet["humi"]
            elif p_type == 'ODOM':
                current_status["pose"]["x"] = packet["pos"]["x"]
                current_status["pose"]["y"] = packet["pos"]["y"]
                current_status["pose"]["theta"] = packet["pos"]["theta"]

        # 3. 5초가 지났을 때만 모아둔 최신 데이터를 클라우드로 1번 전송
        current_time = time.time()
        if current_time - last_publish_time >= PUBLISH_INTERVAL:
            current_status["timestamp"] = int(current_time)
            
            # QoS 0으로 전송 (상태 보고용이므로 가볍게 처리)
            myMQTTClient.publish(publish_topic, json.dumps(current_status), 0)
            
            # 보기 편하게 로그 출력 (원치 않으면 주석 처리)
            print(f"[5초 배치 전송] 미세먼지:{current_status['sensors']['pm25']}, X좌표:{current_status['pose']['x']:.2f}")
            
            last_publish_time = current_time

        # 4. CPU 100% 점유 방지를 위한 미세한 슬립
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n사용자 인터럽트: 시스템을 안전하게 종료합니다.")
    parser.send_stop() # 종료 시 모터 정지
    myMQTTClient.disconnect()
    print("종료 완료")