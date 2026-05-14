import json
import urllib3
import os  # 환경변수를 불러오기 위한 모듈 추가

# HTTP 요청 관리자 생성
http = urllib3.PoolManager()

def lambda_handler(event, context):
    try:
        # 1. 환경변수에서 EC2 IP 주소 꺼내오기
        ec2_ip = os.environ.get('EC2_IP')
        
        # 만약 설정하는 걸 까먹었을 경우를 대비한 방어 코드
        if not ec2_ip:
            raise ValueError("환경변수 'EC2_IP'가 설정되지 않았습니다!")

        # 2. URL 조립
        ec2_url = f"http://{ec2_ip}:3000/api/alert"
        
        print(f"Target URL: {ec2_url}")
        print("IoT Core 이벤트 수신:", event)
        
        # 3. 데이터 인코딩 (JSON -> Bytes)
        encoded_data = json.dumps(event).encode('utf-8')
        
        # 4. EC2로 POST 요청 발사
        response = http.request(
            'POST',
            ec2_url,
            body=encoded_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"전송 결과: {response.status}, {response.data.decode('utf-8')}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Notification sent safely!')
        }
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }