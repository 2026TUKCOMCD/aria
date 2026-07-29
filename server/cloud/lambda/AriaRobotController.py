import json
import boto3
import os

# AWS IoT Core와 통신하기 위한 boto3 클라이언트 생성
iot_client = boto3.client('iot-data', region_name='ap-northeast-2')

def lambda_handler(event, context):
    authorizer_context = event.get('requestContext', {}).get('authorizer', {})
    
    user_name = authorizer_context.get('user_name', 'Unknown User')
    authorized_robot_id = authorizer_context.get('robot_id')

    path_parameters = event.get('pathParameters') or {}
    robot_id = path_parameters.get('id')
    
    if not robot_id:
        return {'statusCode': 400, 'body': json.dumps({'message': 'Missing robot id'})}

    if authorized_robot_id and str(robot_id) != str(authorized_robot_id):
        return {
            'statusCode': 403,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'message': '권한이 없는 로봇을 제어하려고 시도했습니다.'})
        }

    try:
        body = json.loads(event.get('body', '{}'))
        command = body.get('command') 
        value = body.get('value')   
        
        desired_state = {}
        
        if command == "POWER":
            if value not in ["ON", "OFF"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for POWER.'})}
            desired_state['power'] = value
            
        elif command == "MODE":
            # [수정]: MODE의 유효한 값으로 "WAIT"를 추가했습니다.
            if value not in ["AUTO", "MANUAL", "WAIT"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for MODE.'})}
            desired_state['mode'] = value

        elif command == "TURBO":
            if value not in ["ON", "OFF"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for TURBO.'})}
            desired_state['turbo'] = value

        elif command == "SLAM":
            if value not in ["ON", "OFF"]:
                return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid value for SLAM.'})}
            desired_state['slam'] = value
            
        # 기존에 있던 elif command == "WAIT": 블록은 삭제되었습니다.

        else:
            return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid command.'})}

        shadow_payload = {"state": {"desired": desired_state}}
        
        iot_client.update_thing_shadow(
            thingName='aria_robot',
            payload=json.dumps(shadow_payload)
        )

        print(f"[{user_name}] 사용자가 로봇[{robot_id}]에 명령({command}:{value})을 내렸습니다.")

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': True,
                'message': 'Command sent',
                'updated': desired_state
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        return {'statusCode': 500, 'body': json.dumps({'message': str(e)})}