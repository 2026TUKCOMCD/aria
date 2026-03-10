import json
import boto3

# AWS IoT Data 클라이언트 생성
iot_client = boto3.client('iot-data', region_name='ap-northeast-2')

def lambda_handler(event, context):
    try:
        # 1. API Gateway 경로에서 {id} 파라미터 추출
        robot_id = event['pathParameters']['id']
        
        # 2. POST 요청의 Body 데이터 추출 (문자열을 JSON 딕셔너리로 변환)
        body = json.loads(event.get('body', '{}'))
        
        # 3. 요구사항에 명시된 MQTT 토픽 생성
        topic = f"aria/{robot_id}/cmd/nav"
        
        # 4. IoT Core로 MQTT Publish (QoS 1 사용)
        response = iot_client.publish(
            topic=topic,
            qos=1,
            payload=json.dumps(body)
        )
        
        # 5. 비동기 처리 특성에 맞게 202 Accepted 반환
        return {
            'statusCode': 202,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # CORS 허용
            },
            'body': json.dumps({
                'message': "명령이 로봇에게 성공적으로 전송되었습니다.",
                'topic': topic,
                'status': 'Accepted'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': "서버 내부 오류가 발생했습니다."})
        }