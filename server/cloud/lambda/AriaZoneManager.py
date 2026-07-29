import json
import psycopg2
import os
import base64

DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')

# DB_PASSWORD / DB_PASS 둘 다 지원
DB_PASSWORD = os.environ.get('DB_PASSWORD') or os.environ.get('DB_PASS')

DB_PORT = os.environ.get('DB_PORT', '5432')


def make_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5
        )
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None


def response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,PUT,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization'
        },
        'body': json.dumps(body, ensure_ascii=False)
    }


def parse_json_field(value):
    """
    DB json/jsonb 값이 dict/list로 오면 그대로 반환,
    문자열로 오면 json.loads 시도.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value

    return value


def lambda_handler(event, context):
    conn = None
    cursor = None

    try:
        http_method = (
            event.get('httpMethod')
            or event.get('requestContext', {}).get('http', {}).get('method')
        )

        # 브라우저 PUT 요청 preflight 대응
        if http_method == 'OPTIONS':
            return response(200, {'message': 'OK'})

        path_parameters = event.get('pathParameters') or {}
        robot_id = path_parameters.get('id')

        if not robot_id:
            return response(400, {'message': 'Missing robot id'})

        conn = make_connection()
        if not conn:
            return response(500, {'message': 'DB connection failed'})

        cursor = conn.cursor()

        # ==========================================
        # 1. GET API: 구역 목록 조회
        # ==========================================
        if http_method == 'GET':
            # 💡 수정: SELECT 절에 air_grade 추가
            query = """
                SELECT 
                    zone_id,
                    zone_name,
                    center_data,
                    area_data,
                    polygon_data,
                    air_score,
                    air_grade
                FROM robot_zones
                WHERE robot_id = %s
                ORDER BY zone_id;
            """

            cursor.execute(query, (robot_id,))
            rows = cursor.fetchall()

            zones_list = []

            for row in rows:
                # 💡 수정: 반환 JSON에 air_grade 매핑 (row[6])
                zones_list.append({
                    "id": row[0],
                    "name": row[1],
                    "center": parse_json_field(row[2]),
                    "area": parse_json_field(row[3]),
                    "polygon": parse_json_field(row[4]),
                    "air_score": row[5],
                    "air_grade": row[6]
                })

            return response(200, {
                "robot_id": robot_id,
                "zones": zones_list
            })

        # ==========================================
        # 2. PUT API: 구역 정보 등록/수정
        #    일부 zones만 보내도 기존 zones 유지
        # ==========================================
        elif http_method == 'PUT':
            raw_body = event.get('body') or '{}'

            if event.get('isBase64Encoded'):
                raw_body = base64.b64decode(raw_body).decode('utf-8')

            body = json.loads(raw_body)
            new_zones = body.get('zones', [])

            if not isinstance(new_zones, list):
                return response(400, {
                    'message': 'zones must be a list'
                })

            print("PUT zones received:", json.dumps(new_zones, ensure_ascii=False))

            # 💡 수정: INSERT 및 UPDATE SET 부분에 air_grade 로직 추가 (%s 개수 8개로 증가)
            upsert_query = """
                INSERT INTO robot_zones 
                (
                    robot_id,
                    zone_id,
                    zone_name,
                    center_data,
                    area_data,
                    polygon_data,
                    air_score,
                    air_grade
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (robot_id, zone_id)
                DO UPDATE SET
                    zone_name = COALESCE(EXCLUDED.zone_name, robot_zones.zone_name),
                    center_data = COALESCE(EXCLUDED.center_data, robot_zones.center_data),
                    area_data = COALESCE(EXCLUDED.area_data, robot_zones.area_data),
                    polygon_data = COALESCE(EXCLUDED.polygon_data, robot_zones.polygon_data),
                    air_score = COALESCE(EXCLUDED.air_score, robot_zones.air_score),
                    air_grade = COALESCE(EXCLUDED.air_grade, robot_zones.air_grade);
            """

            updated_count = 0
            skipped_count = 0

            for zone in new_zones:
                zone_id = zone.get('id')

                if zone_id is None:
                    skipped_count += 1
                    print(f"zone id 없음. skip: {zone}")
                    continue

                zone_name = zone.get('name')
                
                # 💡 수정: JSON 바디에서 air_grade 추출
                air_score = zone.get('air_score')
                air_grade = zone.get('air_grade')

                c_data = json.dumps(zone.get('center'), ensure_ascii=False) if zone.get('center') is not None else None
                a_data = json.dumps(zone.get('area'), ensure_ascii=False) if zone.get('area') is not None else None
                p_data = json.dumps(zone.get('polygon'), ensure_ascii=False) if zone.get('polygon') is not None else None

                # 💡 수정: 파라미터에 air_grade 포함
                cursor.execute(upsert_query, (
                    robot_id,
                    zone_id,
                    zone_name,
                    c_data,
                    a_data,
                    p_data,
                    air_score,
                    air_grade
                ))

                updated_count += 1

            conn.commit()

            print(f"PUT commit success. updated={updated_count}, skipped={skipped_count}")

            return response(200, {
                "success": True,
                "updated_count": updated_count,
                "skipped_count": skipped_count
            })

        else:
            return response(405, {
                'message': f'Method not allowed: {http_method}'
            })

    except Exception as e:
        if conn:
            conn.rollback()

        print(f"Error: {e}")

        return response(500, {
            'error': str(e)
        })

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()