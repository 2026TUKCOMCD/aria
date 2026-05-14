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

        # 4. 해당 로봇(robot_id)에 연결된 Zone 데이터 조회
        zone_query = """
            SELECT zone_id, zone_name, center_data, area_data, polygon_data
            FROM robot_zones
            WHERE robot_id = %s;
        """
        cursor.execute(zone_query, (robot_id,))
        zones_result = cursor.fetchall()

        zones_list = []
        for z in zones_result:
            zones_list.append({
                "id": z[0],
                "name": z[1],
                "center": z[2],
                "area": z[3],
                "polygon": z[4]
            })
        
        # 5. 이슈 #132 조건에 맞게 응답 JSON 조립
        response_body = {
            "robot_id": robot_id,
            "map_name": result[7],
            "image_url": result[0],  # 프론트가 띄울 S3 이미지 주소
            "metadata": {            # 프론트가 좌표를 계산할 메타데이터
                "resolution": result[1],
                "width": result[2],
                "height": result[3],
                "origin": [result[4], result[5], result[6]]
            },
            "zones": zones_list
        }

        # 6. 성공 응답 (CORS 헤더 포함 - 웹앱에서 에러 안 나게)
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