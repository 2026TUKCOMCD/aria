import json
import boto3
import time
import os
import psycopg2 # PostgreSQL / TimescaleDB 접속용 라이브러리

# AWS 서비스 클라이언트 세팅
iot_client = boto3.client('iot-data', region_name='ap-northeast-2')
s3_client = boto3.client('s3')

# 환경 변수에서 DB 접속 정보 가져오기
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432') # 기본 포트 5432
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASS')
MAP_BUCKET = os.environ.get('MAP_BUCKET','aria-map-storage')
LEARNING_BUCKET = os.environ.get('LEARNING_BUCKET','aria-learningdata-storage')

def delete_s3_folder(bucket_name,prefix):
    """
    지정된 버킷에서 특정 prefix(폴더 경로)를 가진 모든 객체를 삭제
    파일이 많을 경우를 대비해 paginatior를 사용
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            # 삭제할 객체 목록(Key) 추출
            objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]

            #한 번에 일괄 삭제 (최대 1000개씩)
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects':objects_to_delete}
                )
                print(f"Deleted {len(objects_to_delete)} items from {bucket_name}/{prefix}")

def lambda_handler(event, context):
    conn = None
    try:
        # 경로 파라미터에서 로봇 ID 추출
        robot_id = event['pathParameters']['id']
        body = json.loads(event.get('body', '{}'))
        target = body.get('target', 'ALL') # 명세서: "ALL", "MAP", "AI" 등

        # ==========================================
        # [A] MAP 데이터 초기화 (target: "ALL" or "MAP")
        # ==========================================
       
        if target in ["ALL","MAP"]:
            # 1. EC2 PostgreSQL (TimescaleDB) 접속
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cursor = conn.cursor()
            
            # 2. 데이터 삭제 쿼리 실행 (SQL 인젝션 방지를 위해 %s 포맷팅 사용)
            delete_query = "DELETE FROM robot_zones WHERE robot_id = %s;"
            cursor.execute(delete_query, (robot_id,))
            delete_query = "DELETE FROM robot_maps WHERE robot_id = %s;"
            cursor.execute(delete_query, (robot_id,))
            
            # 변경사항 DB에 반영(커밋) 후 연결 종료
            conn.commit()
            cursor.close()
            print(f"[{robot_id}] DB Map data deleted.")
            
            # 3. 범용 S3 버킷의 맵 이미지 삭제
            map_prefix = f"maps/{robot_id}/"
            delete_s3_folder(MAP_BUCKET, map_prefix)


        # ==========================================
        # [B] AI 모드 초기화 (target: "ALL" or "AI")
        # ==========================================
        if target in ["ALL", "AI"]:
            # 스크린샷 기준의 폴더 구조로 타겟팅 (weight/base/는 삭제하지 않음)
            prefix_learning = f"robot_id={robot_id}/learning_data/"
            prefix_weight_current = f"robot_id={robot_id}/weight/current/"
            prefix_weight_history = f"robot_id={robot_id}/weight/history/"
            
            delete_s3_folder(LEARNING_BUCKET, prefix_learning)
            delete_s3_folder(LEARNING_BUCKET, prefix_weight_current)
            delete_s3_folder(LEARNING_BUCKET, prefix_weight_history)
            print(f"[{robot_id}] S3 AI learning data & weights deleted.")


        # ==========================================
        # [C] 기기(로봇)로 MQTT 리셋 명령 발송
        # ==========================================
        
        # 요구사항에 맞는 MQTT Payload 생성 및 발송
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
            'statusCode': 202, 
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f"로봇({robot_id}) 초기화({target}) 명령이 접수되었습니다.",
                'topic': topic
            }, ensure_ascii=False)
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
        # 연결이 열려있다면 안전하게 닫기
        if conn is not None:
            conn.close()