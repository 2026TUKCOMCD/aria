import json
import psycopg2
import os

# 환경 변수 설정
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS') or os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432')

def make_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
            connect_timeout=5
        )
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

def lambda_handler(event, context):
    conn = make_connection()
    if not conn:
        return {'statusCode': 500, 'body': 'DB Connection Failed'}

    try:
        cursor = conn.cursor()
        
        # 1. IoT Rule에서 넘겨준 데이터 추출
        robot_id = event.get('robot_id', '1') # Rule에서 추출된 로봇 ID
        zone_name = event.get('current_zone')
        air_score = event.get('air_score')
        air_grade = event.get('air_grade')

        print(f"수신된 데이터: 로봇={robot_id}, 구역={zone_name}, 점수={air_score}, 등급={air_grade}")

        if not zone_name:
            print("Zone 이름이 전달되지 않아 업데이트를 건너뜁니다.")
            return {'statusCode': 400, 'body': 'Missing zone_name'}

        # 2. robot_zones 테이블 업데이트 SQL
        sql = """
            UPDATE robot_zones 
            SET air_score = %s, 
                air_grade = %s
            WHERE robot_id = %s AND zone_name = %s;
        """
        
        cursor.execute(sql, (air_score, air_grade, robot_id, zone_name))
        conn.commit()
        
        updated_rows = cursor.rowcount
        
        # 3. 디버깅용 결과 로그 출력
        if updated_rows > 0:
            print(f"[{robot_id}] '{zone_name}' 구역 공기질 업데이트 완료. (반영된 행: {updated_rows})")
        else:
            print(f"[{robot_id}] '{zone_name}' 구역을 DB에서 찾을 수 없어 업데이트되지 않았습니다.")

        return {'statusCode': 200, 'body': 'Success'}

    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
        return {'statusCode': 500, 'body': str(e)}
    
    finally:
        if conn:
            cursor.close()
            conn.close()