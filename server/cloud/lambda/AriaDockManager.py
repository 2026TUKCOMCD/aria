import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from decimal import Decimal

# 람다 환경 변수에서 DB 접속 정보 가져오기
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "aria_lambda")
DB_PASS = os.environ.get("DB_PASS")

# datetime 객체를 JSON으로 직렬화하기 위한 헬퍼 함수
def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def lambda_handler(event, context):
    conn = None
    cursor = None
    
    try:
        # HTTP 메서드 및 URL 경로 파라미터 파싱
        method = event.get('requestContext', {}).get('http', {}).get('method')
        path_parameters = event.get('pathParameters', {})
        robot_id = path_parameters.get('id')

        if not robot_id:
            return {
                "statusCode": 400, 
                "body": json.dumps({"success": False, "error": "we need robot ID"})
            }

        # DB 연결
        conn = psycopg2.connect(
            host=DB_HOST, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASS,
            connect_timeout=5 
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # ----------------------------------------
        # [GET] 충전기 위치 조회 API
        # ----------------------------------------
        if method == 'GET':
            query = """
                SELECT robot_id, x, y, theta, updated_at 
                FROM robot_docks 
                WHERE robot_id = %s
            """
            cursor.execute(query, (robot_id,))
            dock_info = cursor.fetchone()

            if dock_info:
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({
                        "success": True, 
                        "data": dock_info
                    }, default=json_serial)
                }
            else:
                return {
                    "statusCode": 404,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({
                        "success": False, 
                        "error": "can't find position of a charger"
                    })
                }

        # ----------------------------------------
        # [POST] 충전기 위치 저장/수정 API (UPSERT)
        # ----------------------------------------
        elif method == 'POST':
            body = json.loads(event.get('body', '{}'))
            x = body.get('x')
            y = body.get('y')
            theta = body.get('theta', 0) # theta 값이 안 들어오면 기본값 0

            if x is None or y is None:
                return {
                    "statusCode": 400, 
                    "body": json.dumps({"success": False, "error": "need x, y position"})
                }

            # ON CONFLICT 구문을 활용하여 데이터가 없으면 INSERT, 있으면 UPDATE
            query = """
                INSERT INTO robot_docks (robot_id, x, y, theta, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (robot_id) 
                DO UPDATE SET 
                    x = EXCLUDED.x,
                    y = EXCLUDED.y,
                    theta = EXCLUDED.theta,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING robot_id, x, y, theta, updated_at;
            """
            cursor.execute(query, (robot_id, x, y, theta))
            conn.commit()
            
            updated_dock = cursor.fetchone()

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": True, 
                    "message": "finish updating", 
                    "data": updated_dock
                }, default=json_serial)
            }

        # 허용되지 않은 메서드 차단
        else:
            return {
                "statusCode": 405, 
                "body": json.dumps({"success": False, "error": "http api do not permited"})
            }

    except Exception as e:
        print(f"Dock Manager Lambda Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"success": False, "error": "Internal server error"})
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()