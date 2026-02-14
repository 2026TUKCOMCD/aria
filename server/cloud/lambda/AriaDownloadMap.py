import json
import psycopg2
import os

# 환경변수에서 DB 연결 정보 가져오기
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_PORT = os.environ.get('DB_PORT', '5432')

def make_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"ERROR: DB 연결 실패 - {e}")
        return None

def lambda_handler(event, context):
    # 1. API Gateway 경로 파라미터에서 로봇 ID 추출 (/robots/{id}/map)
    path_parameters = event.get('pathParameters', {})
    robot_id = path_parameters.get('id')

    if not robot_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Missing robot id in path'})
        }

    # 2. DB 연결
    conn = make_connection()
    if conn is None:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Database connection failed'})
        }

    try:
        cursor = conn.cursor()
        
        # 3. 해당 로봇의 가장 최신 맵 1개 조회 (생성일 기준 내림차순)
        query = """
            SELECT s3_url, resolution, width, height, origin_x, origin_y, origin_theta, map_name
            FROM robot_maps 
            WHERE robot_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1;
        """
        cursor.execute(query, (robot_id,))
        result = cursor.fetchone()

        # 지도가 하나도 없을 경우 (404 Not Found)
        if result is None:
            return {
                'statusCode': 404,
                'body': json.dumps({'message': f'No maps found for robot {robot_id}'})
            }

        # 4. 이슈 #132 조건에 맞게 응답 JSON 조립
        response_body = {
            "robot_id": robot_id,
            "map_name": result[7],
            "image_url": result[0],  # 프론트가 띄울 S3 이미지 주소
            "metadata": {            # 프론트가 좌표를 계산할 메타데이터
                "resolution": result[1],
                "width": result[2],
                "height": result[3],
                "origin": [result[4], result[5], result[6]]
            }
        }

        # 5. 성공 응답 (CORS 헤더 포함 - 웹앱에서 에러 안 나게)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' 
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        print(f"ERROR: DB 조회 실패 - {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error'})
        }
    finally:
        if conn:
            cursor.close()
            conn.close()