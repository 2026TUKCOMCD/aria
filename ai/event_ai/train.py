import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import psycopg2
import os
from sklearn.preprocessing import StandardScaler

# 1. 메타데이터와 로그를 조인하여 로드
def load_advanced_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        database=os.getenv("DB_NAME", "aria"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD")
    )
    # 로그 데이터와 세션 메타데이터를 JOIN으로 합칩니다.
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

# 2. 입력을 10개로 확장한 신경망 모델
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
    # 입력 데이터 칼럼 지정
    feature_cols = [
        'temperature', 'humidity', 'pm25', 'voc',
        'pm25_slope', 'temp_hum_corr', 'pm_voc_corr', 
        'pm25_std', 'voc_std', 'pm25_range'
    ]
    X = df[feature_cols].values
    y = df['final_label'].values.reshape(-1, 1)

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AdvancedCookingDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("고도화된 모델 학습 시작...")
    model.train()
    for epoch in range(15): # 정보가 많으니 15회 정도 돌려봅니다.
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/15, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "event_model.pt")
    joblib.dump(scaler, "scaler.pkl")
    print("고도화 모델 및 스케일러 저장 완료")

if __name__ == "__main__":
    train_advanced()