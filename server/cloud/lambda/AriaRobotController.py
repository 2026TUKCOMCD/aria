import json
import boto3

# AWS IoT Core와 통신하기 위한 boto3 클라이언트 생성
iot_client = boto3.client('iot-data')

def lambda_handler(event, context):
    # 1. URL에서 로봇 ID 추출
    path_parameters = event.get('pathParameters') or {}
    robot_id = path_parameters.get('id')
    
    if not robot_id:
        return {'statusCode': 400, 'body': json.dumps({'message': 'Missing robot id'})}

    # 2. 사용자 정보 추출 (Cognito)
    try:
        claims = event.get('requestContext', {}).get('authorizer', {}).get('jwt', {}).get('claims', {})
        user_email = claims.get('email', 'UnknownUser')
    except (KeyError, TypeError):
        user_email = "UnknownUser"

    try:
        # 3. 프론트엔드 요청 바디 파싱
        body = json.loads(event.get('body', '{}'))
        command = body.get('command') # 예: "POWER", "MODE", "TURBO"
        value = body.get('value')     # 예: "ON", "AUTO", "OFF"
        
        desired_state = {}
        
        # 4. [업데이트된 명세] 유효성 검증 (Validation)
        
        # Case A: 전원 제어
        if command == "POWER":
            if value not in ["ON", "OFF"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for POWER. Must be ON or OFF.'})}
            desired_state['power'] = value
            
        # Case B: 모드 제어 (이제 TURBO가 여기서 빠짐!)
        elif command == "MODE": # 또는 "SET_MODE" (프론트랑 맞추시면 됩니다)
            if value not in ["AUTO", "MANUAL"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for MODE. Must be AUTO or MANUAL.'})}
            desired_state['mode'] = value

        # Case C: [NEW] 터보 제어 (독립된 키)
        elif command == "TURBO":
            if value not in ["ON", "OFF"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for TURBO. Must be ON or OFF.'})}
            desired_state['turbo'] = value
            
        else:
            return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid command. Must be POWER, MODE, or TURBO.'})}

        # 5. AWS IoT Core Shadow 업데이트
        shadow_payload = {"state": {"desired": desired_state}}
        
        iot_client.update_thing_shadow(
            thingName=robot_id,
            payload=json.dumps(shadow_payload)
        )

        # 6. 로그 기록
        print(f"[{user_email}] -> [{robot_id}] 명령: {command}={value}")

        # 7. 응답 반환
        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'success': True,
                'message': 'Command sent',
                'updated': desired_state
            })
        }

    except json.JSONDecodeError:
        return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid JSON format'})}
    except Exception as e:
        print(f"Error: {e}")
        return {'statusCode': 500, 'body': json.dumps({'message': 'Internal server error'})}