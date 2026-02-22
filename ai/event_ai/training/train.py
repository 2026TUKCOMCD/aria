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

# [1] 경로 및 환경 변수 설정
current_file_path = os.path.abspath(__file__)
training_dir = os.path.dirname(current_file_path)
event_ai_dir = os.path.dirname(training_dir)
aria_root = os.path.dirname(os.path.dirname(event_ai_dir))

dotenv_path = os.path.join(aria_root, ".env")
load_dotenv(dotenv_path)

def load_advanced_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        database=os.getenv("DB_NAME", "aria"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD", "1234"),
        port=os.getenv("DB_PORT", "5432")
    )
    
    # [수정 포인트] 로그를 단순히 JOIN하지 않고, 세션별로 평균값을 내서 1줄로 요약합니다.
    # 이렇게 해야 18,000개의 중복이 사라지고 진짜 100개의 학습 데이터가 됩니다.
    query = """
        SELECT 
            AVG(l.temperature) as temperature, 
            AVG(l.humidity) as humidity, 
            AVG(l.pm25) as pm25, 
            AVG(l.voc) as voc, 
            s.pm25_slope, s.temp_hum_corr, s.pm_voc_corr, 
            s.pm25_std, s.voc_std, s.pm25_range,
            s.final_label
        FROM sensor_sessions s
        JOIN sensor_data_logs l ON s.session_id = l.session_id
        GROUP BY s.session_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# [2] 신경망 모델 정의 (성국님의 구조 유지)
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

def train_advanced():
    # GPU(GTX 1650) 사용 가능 여부 체크
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device} (10개 특성 활용)")

    df = load_advanced_data()
    feature_cols = [
        'temperature', 'humidity', 'pm25', 'voc',
        'pm25_slope', 'temp_hum_corr', 'pm_voc_corr', 
        'pm25_std', 'voc_std', 'pm25_range'
    ]
    
    X = df[feature_cols].values
    y = df['final_label'].values.reshape(-1, 1)
    
    print(f"📊 실제 학습 세션 개수: {len(df)}개")
    if len(df) < 10:
        print("⚠️ 데이터가 너무 적습니다. augmenter.py에서 multiply 값을 키워주세요.")

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 텐서 변환
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=min(32, len(df)), shuffle=True)

    # 모델 초기화
    model = AdvancedCookingDetector().to(device)
    criterion = nn.BCELoss()
    # 학습률을 0.01에서 0.001로 낮춰 더 세밀하게 학습하도록 조정
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("📈 고도화된 모델 학습 시작...")
    model.train()
    for epoch in range(25): # 조금 더 세밀하게 25회 학습
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/25, Loss: {total_loss/len(loader):.6f}")

    # [3] 모델 및 스케일러 저장
    models_dir = os.path.join(event_ai_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_path = os.path.join(models_dir, "event_model.pt")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print("\n" + "="*50)
    print(f"✅ 학습 완료! 모델이 성공적으로 갱신되었습니다.")
    print(f"저장 경로: {model_path}")
    print("="*50)

if __name__ == "__main__":
    train_advanced()