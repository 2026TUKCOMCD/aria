import os
import subprocess
from datetime import datetime

# 기존 버킷 이름 유지
BUCKET_NAME = "aria-learningdata-storage"
ROBOT_ID = os.getenv("ROBOT_ID", "robot_id=1") 

def deploy():
    model_path = "refined_model.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 로봇 ID 폴더 내 deploy 경로 설정
    s3_history = f"s3://{BUCKET_NAME}/{ROBOT_ID}/deploy/history/model_{timestamp}.pt"
    s3_latest = f"s3://{BUCKET_NAME}/{ROBOT_ID}/deploy/event_model_latest.pt"

    if os.path.exists(model_path):
        print(f"[{ROBOT_ID}] 모델 배포 시작...")
        # 히스토리 보관 및 최신 모델 갱신
        subprocess.run(["aws", "s3", "cp", model_path, s3_history])
        subprocess.run(["aws", "s3", "cp", model_path, s3_latest])
        print(f"배포 완료: {s3_latest}")
    else:
        print("배포할 모델 파일이 없습니다.")

if __name__ == "__main__":
    deploy()