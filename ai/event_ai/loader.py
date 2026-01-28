import json
import psycopg2
from psycopg2.extras import execute_values
import os  # 환경 변수를 읽기 위한 모듈 추가

def load_data():
    try:
        # 1. 환경 변수에서 DB 접속 정보 가져오기
        db_config = {
            "host": os.getenv("DB_HOST", "db"),
            "database": os.getenv("DB_NAME", "aria"),    
            "user": os.getenv("DB_USER", "user"),        
            "password": os.getenv("DB_PASSWORD", ""), 
            "port": os.getenv("DB_PORT", "5432")
        }

        print(f"'{db_config['host']}' 컨테이너에 접속 시도 중... (DB: {db_config['database']})")
        conn = psycopg2.connect(**db_config, connect_timeout=5)
        cur = conn.cursor()
        print("DB 연결 성공!")

        # 2. JSON 데이터 로드
        with open("aria_augmented_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"데이터 {len(data)}개 주입 시작...")

        for session in data:
            meta = session['meta']
            logs = session['logs']

            cur.execute("""
                INSERT INTO sensor_sessions (
                    pm25_slope, temp_hum_corr, pm_voc_corr, pm25_std, voc_std, pm25_range, final_label
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING session_id;
            """, (meta['pm25_slope'], meta['temp_hum_corr'], meta['pm_voc_corr'], 
                  meta['pm25_std'], meta['voc_std'], meta['pm25_range'], meta['final_label']))
            
            new_id = cur.fetchone()[0]

            log_values = [
                (new_id, log['measured_at'], log['temperature'], 
                 log['humidity'], log['pm25'], log['voc']) for log in logs
            ]
            
            execute_values(cur, """
                INSERT INTO sensor_data_logs (session_id, measured_at, temperature, humidity, pm25, voc)
                VALUES %s;
            """, log_values)

        conn.commit()
        print(f"성공! 모든 데이터가 '{db_config['database']}' DB에 적재되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()

if __name__ == "__main__":
    load_data()