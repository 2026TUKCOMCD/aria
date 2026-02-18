import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import joblib
import numpy as np

# 1. 기존 AdvancedCookingDetector 구조 유지
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

# 2. S3 데이터 로더 (10개 특성 추출 로직 포함)
class AriaS3Dataset(Dataset):
    def __init__(self, manifest_file="ARIA_Sync/valid_manifest.json"):
        with open(manifest_file, "r", encoding="utf-8") as f:
            self.file_paths = json.load(f)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # [중요] 엣지에서 전송한 raw_logs와 stats를 10개 피처로 재조합
        meta = data['meta']
        stats = meta['stats']
        last_log = data['raw_logs'][-1]
        
        # train.py의 feature_cols 순서와 동일하게 매핑
        features = [
            last_log['temperature'], last_log['humidity'], last_log['pm25'], last_log['voc'], # 4개
            stats['pm25_slope'], 0.0, 0.0, # slope + corr(0.0 패딩)
            stats['pm25_std'], stats['voc_std'], stats['pm25_range'] # 나머지 stats
        ]
        
        label = 1 if meta['yolo_verified'] else 0
        return torch.tensor(features, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

def run_retraining():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"재학습 시작 (장치: {device})")

    # 기존 자산 로드
    model_path = "event_model.pt"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path):
        print("기존 모델 파일(event_model.pt)이 없습니다.")
        return

    # 모델 생성 및 가중치 로드
    model = AdvancedCookingDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    scaler = joblib.load(scaler_path)
    
    # 데이터셋 준비
    dataset = AriaS3Dataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            # 기존 스케일러로 정규화 수행
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

    # 결과 저장 (새로운 버전으로 저장하여 기존 모델 보호)
    torch.save(model.state_dict(), "refined_model.pt")
    print("refined_model.pt가 생성되었습니다.")

if __name__ == "__main__":
    run_retraining()