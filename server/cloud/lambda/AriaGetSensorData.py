import json
import psycopg2
import os
from datetime import datetime

# 환경 변수 로드
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')

def make_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME, connect_timeout=5
        )
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def lambda_handler(event, context):
    conn = None
    try:
        # 1. 파라미터 파싱
        path_params = event.get('pathParameters') or {}
        robot_id = path_params.get('id', 'ARIA-001') # 기본값 설정

        # 2. SQL 쿼리 작성 (가장 최신 상태 1개만 조회)
        sql = """
            SELECT 
                battery, is_charging, power_status, operation_mode, current_zone,
                air_score, air_grade,
                pm25, voc, temperature, humidity,
                time
            FROM robot_status_log 
            WHERE robot_id = %s 
            ORDER BY time DESC 
            LIMIT 1
        """
        
        conn = make_connection()
        if not conn:
            raise Exception("DB 접속 실패")
            
        cur = conn.cursor()
        cur.execute(sql, (robot_id,))
        row = cur.fetchone() # 1줄만 가져옴
        
        if not row:
            # 데이터가 없을 경우 빈 응답 처리
            return {
                'statusCode': 404,
                'body': json.dumps({'message': 'No data found for this robot'})
            }

        # 3. JSON 구조 조립 (DB 컬럼 순서대로 매핑)
        # row[0]: battery, row[1]: is_charging ...
        response_data = {
            "robot_status": {
                "battery": row[0],
                "is_charging": row[1],
                "power": row[2],
                "mode": row[3],
                "current_zone": row[4]
            },
            "air_quality": {
                "score": row[5],
                "grade": row[6],
                "sensors": {
                    "pm25": row[7],
                    "voc": row[8],
                    "temperature": row[9],
                    "humidity": row[10]
                }
            },
            "last_updated": str(row[11]) # 마지막 업데이트 시간 추가
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # CORS 허용
            },
            'body': json.dumps(response_data, ensure_ascii=False)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
    finally:
        if conn: conn.close()