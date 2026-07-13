import json
import os
import psycopg2

# ---------------------------------------------------------
# AWS Lambda 핸들러: 프론트엔드에서 토큰 저장 요청이 오면 실행됩니다.
# ---------------------------------------------------------
def lambda_handler(event, context):
    try:
        # 1. 프론트엔드에서 보낸 JSON 데이터 파싱
        body = json.loads(event.get('body', '{}'))
        
        # 프론트엔드에서 넘겨주는 데이터 받기
        qr_token = body.get('qr_token')
        robot_id = body.get('robot_id')
        user_name = body.get('user_name', '알수없음')  # 값이 안 넘어오면 기본값 처리
        robot_name = body.get('robot_name', 'ARIA_ROBOT')
        
        # 필수 데이터 누락 체크
        if not qr_token or not robot_id:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'qr_token과 robot_id는 필수입니다.'})
            }
            
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': '잘못된 JSON 형식입니다.'})
        }

    # 2. PostgreSQL DB 연결 및 데이터 저장
    conn = None
    cursor = None
    try:
        # 람다 환경 변수(Environment Variables)에서 DB 접속 정보 가져오기
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASS'),
            port=os.environ.get('DB_PORT', 5432)
        )
        
        cursor = conn.cursor()
        
        # is_valid는 기본적으로 True(t)로 활성화 상태로 저장
        insert_query = """
            INSERT INTO aria_qr_tokens (qr_token, robot_id, user_name, robot_name, is_valid)
            VALUES (%s, %s, %s, %s, TRUE)
        """
        
        # 쿼리 실행
        cursor.execute(insert_query, (qr_token, robot_id, user_name, robot_name))
        
        # 변경사항 DB에 반영 (필수!)
        conn.commit()
        
        # 3. 프론트엔드에 성공 응답 보내기
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',  # CORS 에러 방지
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'message': 'success',
                'qr_token': qr_token
            })
        }

    except psycopg2.Error as e:
        print(f"DB Error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': '데이터베이스 저장 중 오류가 발생했습니다.'})
        }
        
    finally:
        # 리소스 반환 (메모리 누수 방지)
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
