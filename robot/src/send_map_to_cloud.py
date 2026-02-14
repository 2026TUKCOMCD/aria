import base64
import requests # 터미널에서 pip install requests 필요

# 1. 내 컴퓨터에 있는 진짜 사진 파일 이름
IMAGE_FILE_PATH = "test_map.png" 

# 2. API Gateway 주소 입력
API_URL = "https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com/robots/robot_test_02/map"

def upload_map():
    try:
        # 1. 사진 파일을 '읽기 전용 바이너리(rb)' 모드로 열기
        with open(IMAGE_FILE_PATH, "rb") as image_file:
            # 2. 사진을 통째로 읽어서 Base64 텍스트로 변환하기
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"사진을 Base64로 변환했습니다! (길이: {len(encoded_string)}자)")

        # 3. API로 보낼 JSON 데이터 포장하기
        payload = {
            "image_base64": encoded_string,
            "map_name": "my_real_room_map",
            "metadata": {
                "resolution": 0.05,
                "width": 800,
                "height": 600,
                "origin": [-10.5, -5.2, 0.0]
            }
        }

        # 4. 서버(API Gateway)로 POST 요청 쏘기!
        print("서버로 전송 중...")
        response = requests.post(
            API_URL, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )

        # 5. 결과 확인
        print(f"상태 코드: {response.status_code}")
        print(f"서버 응답: {response.json()}")

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    upload_map()