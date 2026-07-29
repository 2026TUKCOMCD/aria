import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# 환경변수에서 DB 연결 정보 로드 (값이 없을 경우를 대비한 기본값 설정 가능)
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME", "postgres") # 기본값 postgres
DB_USER = os.environ.get("DB_USER", "aria_lambda")
DB_PASS = os.environ.get("DB_PASS")

def api_verify_token(request_headers):
    conn = None
    try:
        # 1. Header에서 토큰 추출
        token = request_headers.get('Authorization') or request_headers.get('authorization')
        
        if not token:
            return build_response(400, {"valid": False, "message": "토큰이 누락되었습니다."})

        # "Bearer 랜덤토큰" 형태 처리
        if token.startswith("Bearer "):
            token = token.split(" ")[1]

        # 2. PostgreSQL DB 연결 (네트워크 타임아웃 5초 설정)
        conn = psycopg2.connect(
            host=DB_HOST, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASS,
            connect_timeout=5 
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 3. SQL 쿼리 실행
        query = """
            SELECT robot_id, user_name, robot_name, is_valid 
            FROM aria_qr_tokens 
            WHERE qr_token = %s
        """
        cursor.execute(query, (token,))
        item = cursor.fetchone()

        # 4. 검증 및 응답 데이터 조립
        if item:
            if item['is_valid']:
                return build_response(200, {
                    "valid": True,
                    "robot_id": item['robot_id'],
                    "user_name": item['user_name'],
                    "robot_name": item['robot_name']
                })
            else:
                return build_response(401, {"valid": False, "message": "이미 사용되거나 만료된 토큰입니다."})
        else:
            return build_response(404, {"valid": False, "message": "유효하지 않은 QR 코드입니다."})

    except Exception as e:
        print(f"DB Error: {e}")
        return build_response(500, {"valid": False, "message": "내부 서버 오류가 발생했습니다."})
        
    finally:
        # DB 연결 자원 해제
        if conn:
            cursor.close()
            conn.close()

def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*" # 웹앱(프론트엔드) 통신을 위한 CORS 허용
        },
        "body": json.dumps(body)
    }