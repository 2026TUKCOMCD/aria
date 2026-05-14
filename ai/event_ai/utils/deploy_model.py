import os
import subprocess
from datetime import datetime
import shutil

BUCKET_NAME = "aria-learningdata-storage"
ROBOT_ID = os.getenv("ROBOT_ID", "robot_id=1") 

def deploy():
    model_file = "refined_model.pt"
    scaler_file = "refined_scaler.pkl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # S3 세부 경로 설정
    s3_current = f"s3://{BUCKET_NAME}/{ROBOT_ID}/weight/current"
    s3_history = f"s3://{BUCKET_NAME}/{ROBOT_ID}/weight/history"

    if os.path.exists(model_file):
        print(f"[{ROBOT_ID}] S3 배포 및 이력 저장 시작...")
        
        # 1. current 업데이트 (항상 최신본 유지)
        subprocess.run(["aws", "s3", "cp", model_file, f"{s3_current}/event_model_latest.pt"])
        subprocess.run(["aws", "s3", "cp", scaler_file, f"{s3_current}/scaler.pkl"])
        
        # 2. history 누적 (학습할 때마다 쌓임)
        subprocess.run(["aws", "s3", "cp", model_file, f"{s3_history}/model_{timestamp}.pt"])
        subprocess.run(["aws", "s3", "cp", scaler_file, f"{s3_history}/scaler_{timestamp}.pkl"])
        
        print(f"배포 및 이력 저장 완료: {s3_current}, {s3_history}")
        
        # 임시파일 정리
        os.remove(model_file)
        os.remove(scaler_file)

if __name__ == "__main__":
    deploy()