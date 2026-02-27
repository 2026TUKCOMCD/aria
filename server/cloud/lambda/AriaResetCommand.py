import json
import boto3
import time
import os
import psycopg2 # PostgreSQL / TimescaleDB 접속용 라이브러리

# AWS 서비스 클라이언트 세팅
iot_client = boto3.client('iot-data', region_name='ap-northeast-2')

# 환경 변수에서 DB 접속 정보 가져오기
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432') # 기본 포트 5432
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASS')

def lambda_handler(event, context):
    conn = None
    try:
        # 1. 경로 파라미터에서 로봇 ID 추출
        robot_id = event['pathParameters']['id']
        
        # 2. EC2 PostgreSQL (TimescaleDB) 접속
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # 3. 데이터 삭제 쿼리 실행 (SQL 인젝션 방지를 위해 %s 포맷팅 사용)
        # 'robot_data' 부분은 민재 님의 실제 테이블명으로 꼭 바꿔주세요!
        delete_query = "DELETE FROM robot_zones WHERE robot_id = %s;"
        cursor.execute(delete_query, (robot_id,))
        delete_query = "DELETE FROM robot_maps WHERE robot_id = %s;"
        cursor.execute(delete_query, (robot_id,))
        
        # 변경사항 DB에 반영(커밋) 후 연결 종료
        conn.commit()
        cursor.close()
        
        # 4. 요구사항에 맞는 MQTT Payload 생성 및 발송
        topic = f"aria/{robot_id}/cmd/control"
        payload = {
            "target": "RESET",
            "action": "TURN_ON",
            "timestamp": int(time.time())
        }
        
        iot_client.publish(
            topic=topic,
            qos=1,
            payload=json.dumps(payload)
        )
        
        return {
            'statusCode': 200, 
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f"로봇({robot_id}) 초기화 명령 및 TimescaleDB 정리가 완료되었습니다.",
                'topic': topic
            })
        }
        
    except psycopg2.Error as db_err:
        print(f"DB Error: {str(db_err)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': "데이터베이스 처리 중 오류가 발생했습니다."})
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': "서버 내부 오류가 발생했습니다."})
        }
    finally:
        # DB 연결이 열려있다면 안전하게 닫기
        if conn is not None:
            conn.close()