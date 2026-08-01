import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os

DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "aria_lambda")
DB_PASS = os.environ.get("DB_PASS")

def lambda_handler(event, context):
    try:
        # 1. Payload v2.0에서는 토큰이 headers 안에 소문자로 들어옵니다.
        headers = event.get('headers', {})
        auth_header = headers.get('authorization', '') 
        
        # 토큰이 없으면 즉시 거부 (isAuthorized: False)
        if not auth_header:
            return {"isAuthorized": False}

        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            token = auth_header

        # 2. DB 연결
        conn = psycopg2.connect(
            host=DB_HOST, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASS,
            connect_timeout=5 
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 3. 쿼리 실행
        query = """
            SELECT robot_id, user_name, robot_name, is_valid 
            FROM aria_qr_tokens 
            WHERE qr_token = %s
        """
        cursor.execute(query, (token,))
        item = cursor.fetchone()

        # 4. 검증 및 '단순' 응답 반환 (스크린샷 설정에 완벽히 호환)
        if item and item['is_valid']:
            # 성공 시: true 반환 및 ARIA 로봇 제어 람다로 정보 넘기기
            return {
                "isAuthorized": True,
                "context": {
                    "robot_id": str(item['robot_id']),
                    "user_name": str(item['user_name']),
                    "robot_name": str(item['robot_name'])
                }
            }
        else:
            # 실패 시: false 반환 (자동으로 403 Forbidden 처리됨)
            return {"isAuthorized": False}

    except Exception as e:
        print(f"Authorizer Error: {e}")
        # DB 에러 등 서버 내부 문제 시에도 안전하게 접근 차단
        return {"isAuthorized": False}
        
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()