import os
import subprocess

# 설정
BUCKET_NAME = "aria-learningdata-storage"
ROBOT_ID = "robot_id=1"
LOCAL_BASE_DIR = "../models/"  # 로컬 초기 파일 경로

def upload_base_assets():
    model_file = os.path.join(LOCAL_BASE_DIR, "event_model.pt")
    scaler_file = os.path.join(LOCAL_BASE_DIR, "scaler.pkl")
    
    # S3 목표 경로
    s3_base_path = f"s3://{BUCKET_NAME}/{ROBOT_ID}/weight/base"

    print(f"[{ROBOT_ID}] 초기 모델 자산을 S3 {s3_base_path}에 업로드 중...")
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        # 정확히 파일명을 지정해서 업로드 (상위 디렉토리 삭제 위험 방지)
        subprocess.run(["aws", "s3", "cp", model_file, f"{s3_base_path}/event_model.pt"])
        subprocess.run(["aws", "s3", "cp", scaler_file, f"{s3_base_path}/scaler.pkl"])
        print("업로드 완료!")
    else:
        print(f"오류: 로컬 {LOCAL_BASE_DIR} 폴더에 파일이 없습니다.")

if __name__ == "__main__":
    upload_base_assets()