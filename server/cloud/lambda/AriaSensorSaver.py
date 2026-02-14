import json
import psycopg2
import os
import sys

# 환경 변수 설정 (Lambda Configuration에서 설정 필요)
host = os.environ.get('DB_HOST')
dbname = os.environ.get('DB_NAME')
user = os.environ.get('DB_USER')
password = os.environ.get('DB_PASS')
port = os.environ.get('DB_PORT', "5432")

def make_connection():
    try:
        conn = psycopg2.connect(
            host=host,
            database=dbname,
            user=user,
            password=password,
            port=port
        )
        return conn
    except Exception as e:
        print(f"ERROR: DB 연결 실패 - {e}")
        return None

def lambda_handler(event, context):
    """
    IoT Rule에서 전처리된 데이터를 받아 robot_status_log 테이블에 저장
    """
    conn = make_connection()
    
    if conn is None:
        return {'statusCode': 500, 'body': json.dumps('Database Connection Failed')}

    try:
        cursor = conn.cursor()
        
        # 1. 들어온 데이터 확인 (CloudWatch 로그용)
        # IoT Rule이 SQL로 다듬어서 보낸 데이터가 여기 들어옴
        print(" Processed Event:", event)
        
        # 2. 데이터 추출
        # IoT Rule SQL에서 이미 이름을 맞춰줬으므로 event.get()으로 바로 꺼냄
        robot_id = event.get('device_id', 'unknown_robot')
        server_timestamp = event.get('server_timestamp') # Unix Timestamp (ms 단위일 수 있음)
        
        # 상태 정보
        battery = event.get('battery', 0)
        is_charging = event.get('is_charging', False)
        power_status = event.get('power_status', 'OFF')
        operation_mode = event.get('operation_mode', 'AUTO')
        
        # 센서 데이터 (Rule에서 Flatten 되었음)
        temperature = event.get('temperature', 0.0)
        humidity = event.get('humidity', 0.0)
        pm25 = event.get('pm25', 0.0)
        voc = event.get('voc', 0) # DB 스키마에 맞춰 INT/FLOAT 확인 (init.sql에선 INT였음)
        
        # 공기질 점수
        air_score = event.get('air_score', 0)
        air_grade = event.get('air_grade', 'UNKNOWN')

        # 3. SQL 쿼리 작성 (robot_status_log 테이블)
        # time 컬럼에는 to_timestamp()를 사용하여 유닉스 타임스탬프를 DB 시간 포맷으로 변환
        sql = """
            INSERT INTO robot_status_log (
                time, robot_id, 
                battery, is_charging, power_status, operation_mode, 
                temperature, humidity, pm25, voc, 
                air_score, air_grade
            )
            VALUES (
                to_timestamp(%s / 1000.0), %s, 
                %s, %s, %s, %s, 
                %s, %s, %s, %s, 
                %s, %s
            );
        """
        
        # 주의: IoT Core의 timestamp()는 밀리초(ms) 단위일 수 있으므로 
        # 초 단위로 맞추기 위해 1000.0으로 나누는 로직을 넣음
        
        cursor.execute(sql, (
            server_timestamp, robot_id,
            battery, is_charging, power_status, operation_mode,
            temperature, humidity, pm25, voc,
            air_score, air_grade
        ))
        
        conn.commit()
        print(f" Status data saved for {robot_id}")
        
        return {'statusCode': 200, 'body': json.dumps('Status Saved!')}

    except Exception as e:
        print(f" Error: {e}")
        if conn:
            conn.rollback()
        return {'statusCode': 500, 'body': json.dumps(str(e))}
    
    finally:
        if conn:
            cursor.close()
            conn.close()