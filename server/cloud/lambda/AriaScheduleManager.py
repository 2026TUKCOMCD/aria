import json
import boto3
import os
import psycopg2
from datetime import datetime

# 환경 변수 
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_PORT = os.environ.get('DB_PORT', '5432')

SCHEDULER_ROLE_ARN = os.environ.get('SCHEDULER_ROLE_ARN') 

# 클라이언트 초기화
scheduler_client = boto3.client('scheduler', region_name='ap-northeast-2')
iot_client = boto3.client('iot-data', region_name='ap-northeast-2')

def make_connection():
    try:
        return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

def lambda_handler(event, context):
    # =========================================================
    # [모드 1] 스케줄러가 지정된 시간에 람다를 깨웠을 때 (알람 모드)
    # =========================================================
    if event.get('source') == 'eventbridge_scheduler':
        robot_id = event.get('robot_id')
        action = event.get('action') # 'TURN_ON' 또는 'TURN_OFF'
        
        topic = f"aria/{robot_id}/cmd/control"
        payload = {
            "target": "POWER",
            "action": action,
            "timestamp": int(datetime.now().timestamp())
        }
        
        try:
            iot_client.publish(topic=topic, qos=1, payload=json.dumps(payload))
            print(f"MQTT 성공: {topic} -> {action}")
            return {"statusCode": 200, "body": "MQTT Published Successfully"}
        except Exception as e:
            print(f"MQTT 실패: {e}")
            return {"statusCode": 500, "body": str(e)}

    # =========================================================
    # [모드 2] 웹앱(API Gateway)에서 스케줄을 저장할 때 (API 모드)
    # =========================================================
    conn = None
    try:
        robot_id = event.get('pathParameters', {}).get('id')
        if not robot_id:
            return {'statusCode': 400, 'body': json.dumps({'message': 'Missing robot id'})}

        body = json.loads(event.get('body', '{}'))
        wake_time = body.get('wake_time')   
        sleep_time = body.get('sleep_time') 
        enabled = body.get('enabled', True)
        
        conn = make_connection()
        cursor = conn.cursor()
        
        # 1. DB 저장 (Upsert)
        upsert_query = """
            INSERT INTO robot_schedules (robot_id, wake_time, sleep_time, is_enabled)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (robot_id) DO UPDATE SET 
                wake_time = EXCLUDED.wake_time, sleep_time = EXCLUDED.sleep_time, 
                is_enabled = EXCLUDED.is_enabled, updated_at = NOW();
        """
        cursor.execute(upsert_query, (robot_id, wake_time, sleep_time, enabled))
        conn.commit()
        
        # 2. 로봇에 스케줄 자체를 전달 (retained → 로봇이 재접속해도 마지막 값을 받음)
        #    이전 방식(EventBridge가 정해진 시각에 POWER TURN_OFF/ON 발행)은 로봇이
        #    모터만 한 번 멈추고 AI모드를 유지해서 수면 절전이 실제로 일어나지 않았다.
        #    이제 판단은 로봇 컨트롤러가 로컬 시계로 하므로 네트워크가 끊겨도 동작한다.
        iot_client.publish(
            topic=f"aria/{robot_id}/cmd/schedule",
            qos=1,
            retain=True,
            payload=json.dumps({"sleep_time": sleep_time, "wake_time": wake_time, "enabled": enabled}),
        )

        # 예전 방식으로 만들어 둔 EventBridge 알람이 남아 있으면 정리한다.
        for schedule_type in ('wake', 'sleep'):
            try:
                scheduler_client.delete_schedule(Name=f"aria_{robot_id}_{schedule_type}_schedule")
            except scheduler_client.exceptions.ResourceNotFoundException:
                pass
            except Exception as e:  # 권한(scheduler:DeleteSchedule) 없음 등 — 저장 자체는 성공 처리
                print(f"옛 EventBridge 스케줄 정리 실패(무시): {e}")
            
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"wake_time": wake_time, "sleep_time": sleep_time, "enabled": enabled})
        }
        
    except Exception as e:
        print(f"에러: {e}")
        if conn: conn.rollback()
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    finally:
        if conn:
            cursor.close()
            conn.close()

def upsert_scheduler(robot_id, schedule_type, time_str, action, state, lambda_arn):
    """EventBridge Scheduler를 생성하여 지정된 시간에 이 람다 함수를 다시 호출하게 합니다."""
    hour, minute = time_str.split(':')
    cron_expr = f"cron({minute} {hour} * * ? *)"
    schedule_name = f"aria_{robot_id}_{schedule_type}_schedule"
    
    # 이제 타겟이 IoT Core가 아니라 '이 람다 함수' 입니다.
    target_config = {
        'Arn': lambda_arn, 
        'RoleArn': SCHEDULER_ROLE_ARN,
        'Input': json.dumps({
            "source": "eventbridge_scheduler",
            "robot_id": robot_id,
            "action": action
        })
    }
    
    try:
        scheduler_client.get_schedule(Name=schedule_name)
        scheduler_client.update_schedule(
            Name=schedule_name, ScheduleExpression=cron_expr, ScheduleExpressionTimezone='Asia/Seoul',
            FlexibleTimeWindow={'Mode': 'OFF'}, Target=target_config, State=state
        )
    except scheduler_client.exceptions.ResourceNotFoundException:
        scheduler_client.create_schedule(
            Name=schedule_name, ScheduleExpression=cron_expr, ScheduleExpressionTimezone='Asia/Seoul',
            FlexibleTimeWindow={'Mode': 'OFF'}, Target=target_config, State=state
        )