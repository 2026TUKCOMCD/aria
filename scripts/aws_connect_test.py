# test_connect.py
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import json
import os

# 1. AWS IoT 엔드포인트 (AWS 콘솔 -> 설정 -> 디바이스 데이터 엔드포인트 복사)
ENDPOINT = "adecukeeb0iln-ats.iot.ap-northeast-2.amazonaws.com"

# 2. 클라이언트 ID (원하는 이름 아무거나)
CLIENT_ID = "Macbook_Test_Client"

# 3. 인증서 파일 경로 (파일명이 정확한지 꼭 확인)
PATH_TO_ROOT = "../certs/AmazonRootCA1.pem"
PATH_TO_CERT = "../certs/bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-certificate.pem.crt"
PATH_TO_KEY  = "../certs/bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-private.pem.key"

# ==================================================================

def customCallback(client, userdata, message):
    print(f"\n [수신] 토픽: {message.topic}")
    print(f"   내용: {message.payload.decode('utf-8')}")

# 메인 로직 시작
print("=============================================")
print(f" AWS IoT Core 연결 테스트 (Python 3.10)")
print("=============================================")

# 파일 존재 여부 확인 (실수 방지용)
if not os.path.exists(PATH_TO_CERT) or not os.path.exists(PATH_TO_KEY):
    print(" [오류] 인증서 파일을 찾을 수 없습니다. 파일명과 경로를 확인하세요!")
    print(f"   - 현재 설정된 인증서 경로: {PATH_TO_CERT}")
    print(f"   - 현재 설정된 키 경로: {PATH_TO_KEY}")
    exit()

try:
    # 클라이언트 설정
    myMQTTClient = AWSIoTMQTTClient(CLIENT_ID)
    myMQTTClient.configureEndpoint(ENDPOINT, 8883)
    myMQTTClient.configureCredentials(PATH_TO_ROOT, PATH_TO_KEY, PATH_TO_CERT)

    # 연결 설정 (타임아웃 등 안정성 설정)
    myMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
    myMQTTClient.configureOfflinePublishQueueing(-1)  
    myMQTTClient.configureDrainingFrequency(2) 
    myMQTTClient.configureConnectDisconnectTimeout(10) 
    myMQTTClient.configureMQTTOperationTimeout(5) 

    # 1. 연결 시도 (핸드셰이크)
    print(f" 엔드포인트로 연결 시도 중...")
    if myMQTTClient.connect():
        print(" [성공] AWS IoT Core에 보안 연결(TLS) 완료!")
    else:
        print(" [실패] 연결되지 않았습니다.")
        exit()

    # 2. 메시지 전송 (Publish) 테스트
    topic = "aria/test/topic"
    payload = {
        "message": "Hello from Macbook",
        "timestamp": int(time.time()),
        "status": "Test OK",
        "python_version": "3.10"
    }
    
    print(f" [전송] 토픽: {topic}")
    print(f"   데이터: {json.dumps(payload)}")
    
    myMQTTClient.publish(topic, json.dumps(payload), 1)
    
    print("\n 2초간 대기 후 종료합니다...")
    time.sleep(2)

    # 3. 연결 종료
    myMQTTClient.disconnect()
    print(" 연결 종료 완료")

except Exception as e:
    print(f" [에러 발생] {e}")
