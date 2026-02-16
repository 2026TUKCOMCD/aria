import json
import psycopg2
import os

DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432')

def make_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

def lambda_handler(event, context):
    http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
    path_parameters = event.get('pathParameters') or {}
    robot_id = path_parameters.get('id')

    if not robot_id:
        return {'statusCode': 400, 'body': json.dumps({'message': 'Missing robot id'})}

    conn = make_connection()
    if not conn:
        return {'statusCode': 500, 'body': json.dumps({'message': 'DB connection failed'})}

    try:
        cursor = conn.cursor()

        # ==========================================
        # 1. GET API: 구역 목록 조회
        # ==========================================
        if http_method == 'GET':
            query = "SELECT zone_id, zone_name, center_data, area_data FROM robot_zones WHERE robot_id = %s"
            cursor.execute(query, (robot_id,))
            rows = cursor.fetchall()

            zones_list = []
            for row in rows:
                zones_list.append({
                    "id": row[0],
                    "name": row[1],
                    "center": row[2],  # JSONB라 바로 딕셔너리로 변환됨
                    "area": row[3]
                })

            response_body = {
                "robot_id": robot_id,
                "zones": zones_list
            }

            return {
                'statusCode': 200,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps(response_body, ensure_ascii=False)
            }

        # ==========================================
        # 2. PUT API: 구역 정보 등록/수정 (UPSERT 방식)
        # ==========================================
        elif http_method == 'PUT':
            body = json.loads(event.get('body', '{}'))
            new_zones = body.get('zones', [])

            # 2-1. 이번에 프론트엔드가 보내준 방 이름들만 리스트로 get
            new_zone_names = [zone.get('name') for zone in new_zones]

            # 2-2. 영리한 삭제: 프론트엔드가 안 보낸 방(즉, 앱에서 삭제한 방)만 골라서 delete
            if new_zone_names:
                format_strings = ','.join(['%s'] * len(new_zone_names))
                delete_query = f"DELETE FROM robot_zones WHERE robot_id = %s AND zone_name NOT IN ({format_strings})"
                cursor.execute(delete_query, [robot_id] + new_zone_names)
            else:
                # 빈 배열을 보냈다면 방을 싹 다 지웠다는 뜻
                cursor.execute("DELETE FROM robot_zones WHERE robot_id = %s", (robot_id,))

            # 2-3. UPSERT 쿼리 (PostgreSQL의 필살기 ON CONFLICT 사용)
            upsert_query = """
                INSERT INTO robot_zones (robot_id, zone_name, center_data, area_data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (robot_id, zone_name) 
                DO UPDATE SET 
                    center_data = EXCLUDED.center_data,
                    area_data = EXCLUDED.area_data
            """
            
            for zone in new_zones:
                cursor.execute(upsert_query, (
                    robot_id, 
                    zone.get('name'), 
                    json.dumps(zone.get('center')), 
                    json.dumps(zone.get('area'))
                ))
            
            conn.commit() # 변경사항 확정

            return {
                'statusCode': 200,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({"success": True})
            }

    except Exception as e:
        if conn:
            conn.rollback() # 에러 나면 DB 원상복구
        print(f"Error: {e}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    finally:
        if conn:
            cursor.close()
            conn.close()