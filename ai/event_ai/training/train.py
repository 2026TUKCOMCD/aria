import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import psycopg2
import os
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# [1] 경로 설정 최적화
# 현재 파일 위치: aria/ai/event_ai/training/train.py
current_file_path = os.path.abspath(__file__)
training_dir = os.path.dirname(current_file_path)   # training/
event_ai_dir = os.path.dirname(training_dir)       # event_ai/
ai_dir = os.path.dirname(event_ai_dir)             # ai/
aria_root = os.path.dirname(ai_dir)                # aria/

# aria/.env 파일을 로드하여 DB 접속 정보를 가져옵니다
dotenv_path = os.path.join(aria_root, ".env")
load_dotenv(dotenv_path)

# [2] 메타데이터와 로그를 조인하여 로드
def load_advanced_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        database=os.getenv("DB_NAME", "aria"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )
    query = """
        SELECT 
            l.temperature, l.humidity, l.pm25, l.voc, 
            s.pm25_slope, s.temp_hum_corr, s.pm_voc_corr, 
            s.pm25_std, s.voc_std, s.pm25_range,
            s.final_label
        FROM sensor_data_logs l
        JOIN sensor_sessions s ON l.session_id = s.session_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# [3] 신경망 모델 정의
class AdvancedCookingDetector(nn.Module):
    def __init__(self):
        super(AdvancedCookingDetector, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(10, 32), # 4(기본) + 6(메타) = 10
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def train_advanced():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device} (10개 특성 활용)")

    df = load_advanced_data()
    feature_cols = [
        'temperature', 'humidity', 'pm25', 'voc',
        'pm25_slope', 'temp_hum_corr', 'pm_voc_corr', 
        'pm25_std', 'voc_std', 'pm25_range'
    ]
    X = df[feature_cols].values
    y = df['final_label'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AdvancedCookingDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("📈 고도화된 모델 학습 시작...")
    model.train()
    for epoch in range(15):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/15, Loss: {total_loss/len(loader):.4f}")

    # [4] 모델 저장 경로 수정 (models/ 폴더 내 저장)
    # training/ 폴더에서 한 단계 위인 event_ai/models/ 폴더를 지정합니다.
    models_dir = os.path.join(event_ai_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_path = os.path.join(models_dir, "event_model.pt")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print("\n" + "="*50)
    print(f"학습 완료 및 모델 저장 성공!")
    print(f"모델: {model_path}")
    print(f"스케일러: {scaler_path}")
    print("="*50)

if __name__ == "__main__":
    train_advanced()