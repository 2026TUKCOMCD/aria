import torch
import pandas as pd
import psycopg2
import os

def check_all():
    print("--- ARIA Environment Verification ---")
    
    # [항목 3] GPU 가속 확인
    cuda_ok = torch.cuda.is_available()
    print(f"1. CUDA Available: {cuda_ok}")
    if cuda_ok:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # [항목 4] 데이터 로더 초기 구현 (DB 연동)
    print("2. Database Connection & Data Loading...")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "db"),
            database=os.getenv("DB_NAME", "aria"),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD")
        )
        
        # sensor_data_logs 테이블에서 최신 데이터 읽어오기
        query = "SELECT * FROM sensor_data_logs LIMIT 5;"
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"Success! Loaded {len(df)} rows from DB.")
        print(df.head(2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_all()