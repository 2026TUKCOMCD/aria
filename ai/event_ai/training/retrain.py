import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import joblib
import numpy as np
import subprocess
import shutil

# [설정] S3 및 로봇 정보
BUCKET_NAME = "aria-learningdata-storage"
ROBOT_ID = os.getenv("ROBOT_ID", "robot_id=1")

# 1. 모델 구조 정의 (기존 구조 유지)
class AdvancedCookingDetector(nn.Module):
    def __init__(self):
        super(AdvancedCookingDetector, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(10, 32), 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

# AriaS3Dataset 클래스의 __init__ 부분을 다음과 같이 수정
class AriaS3Dataset(Dataset):
    def __init__(self, manifest_file=None):
        # 실행 파일(retrain.py)의 절대 경로를 기준으로 프로젝트 루트를 찾음
        self.current_file_path = os.path.abspath(__file__)
        self.training_dir = os.path.dirname(self.current_file_path)
        self.project_root = os.path.dirname(self.training_dir)
        
        # 매니페스트 파일 경로가 인자로 안 들어오면 직접 생성
        if manifest_file is None:
            manifest_file = os.path.join(self.project_root, "sync", "valid_manifest.json")
        
        self.sync_dir = os.path.join(self.project_root, "sync")
        
        print(f"매니페스트 확인 중: {manifest_file}") # 경로 디버깅용
        
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"매니페스트 파일이 없습니다: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            self.file_paths = json.load(f)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw_path = self.file_paths[idx]
        filename = os.path.basename(raw_path)
        actual_path = os.path.join(self.sync_dir, "data_lake", filename)

        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"데이터 파일이 없습니다: {actual_path}")

        with open(actual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        meta = data['meta']
        stats = meta['stats']
        last_log = data['raw_logs'][-1]
        
        # 특성 추출 (10개 feature)
        features = [
            last_log['temperature'], last_log['humidity'], last_log['pm25'], last_log['voc'],
            stats['pm25_slope'], 0.0, 0.0,
            stats['pm25_std'], stats['voc_std'], stats['pm25_range']
        ]
        
        label = 1 if meta['yolo_verified'] else 0
        return torch.tensor(features, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

# 3. 자산 준비 함수 (S3 current -> S3 base -> Local 순으로 탐색)
def prepare_assets(model_path, scaler_path):
    s3_weight_root = f"s3://{BUCKET_NAME}/{ROBOT_ID}/weight"
    local_base_dir = "../models/base"

    print(f"[MLOps] 자산 준비 중... (대상: {ROBOT_ID})")
    
    # 1순위: S3 current 확인
    res_curr = subprocess.run(["aws", "s3", "cp", f"{s3_weight_root}/current/event_model_latest.pt", model_path], capture_output=True)
    
    if res_curr.returncode == 0:
        print(">> S3 'current' 디렉토리에서 최신 모델을 성공적으로 로드했습니다.")
        subprocess.run(["aws", "s3", "cp", f"{s3_weight_root}/current/scaler.pkl", scaler_path])
    else:
        # 2순위: S3 base 확인
        print(">> 'current'에 모델이 없습니다. S3 'base'를 확인합니다.")
        res_base = subprocess.run(["aws", "s3", "cp", f"{s3_weight_root}/base/event_model.pt", model_path], capture_output=True)
        
        if res_base.returncode == 0:
            print(">> S3 'base' 디렉토리에서 초기 모델을 로드했습니다.")
            subprocess.run(["aws", "s3", "cp", f"{s3_weight_root}/base/scaler.pkl", scaler_path])
        else:
            # 3순위: 로컬 base 확인 (최후의 보루)
            print(">> S3에 자산이 없습니다. 로컬 'models/base' 폴더를 사용합니다.")
            shutil.copy(os.path.join(local_base_dir, "event_model.pt"), model_path)
            shutil.copy(os.path.join(local_base_dir, "scaler.pkl"), scaler_path)

# 4. 메인 재학습 로직
def run_retraining():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"재학습 시작 (장치: {device})")

    model_dir = "../models"
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, "event_model.pt")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # 가중치 파일 준비
    prepare_assets(model_path, scaler_path)

    # 모델 및 스케일러 로드
    model = AdvancedCookingDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    scaler = joblib.load(scaler_path)
    
    # 데이터셋 로드
    try:
        dataset = AriaS3Dataset()
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            # 스케일링 적용
            inputs_scaled = scaler.transform(inputs.numpy())
            inputs_tensor = torch.FloatTensor(inputs_scaled).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(loader):.4f}")

    # 배포용 임시 파일 저장 (deploy_model.py에서 처리할 수 있도록)
    torch.save(model.state_dict(), "refined_model.pt")
    shutil.copy(scaler_path, "refined_scaler.pkl") 
    print("refined_model.pt 및 refined_scaler.pkl 생성이 완료되었습니다.")

if __name__ == "__main__":
    run_retraining()