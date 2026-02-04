import torch
import torch.nn as nn
import joblib
import os
from collections import deque

# 1. 모델 구조 (train.py에서 가져옴 - 가중치를 불러오기 위해 필요)
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

# 2. 엣지 환경 검증 및 로딩 (check_env.py의 역할 통합)
def init_edge_engine():
    print("--- 🔍 ARIA Edge Inference Engine ---")
    device = torch.device("cpu") # 파이는 CPU 고정
    
    try:
        # 모델 로드
        model = AdvancedCookingDetector()
        model.load_state_dict(torch.load("event_model.pt", map_location=device, weights_only=True))
        model.eval()
        
        # 스케일러 로드
        scaler = joblib.load("scaler.pkl")
        
        print("✅ Model & Scaler loaded successfully on CPU.")
        return model, scaler
    except Exception as e:
        print(f"❌ Loading Error: {e}")
        return None, None

# 3. 실시간 추론 및 조기 감지 로직 (C3-2 반영)
def run_inference():
    model, scaler = init_edge_engine()
    if not model: return

    # C3-1: 30분 센서 버퍼 (10초 단위 샘플링 시 180개)
    buffer = deque(maxlen=180) 
    
    print("🚀 실시간 감시 시작...")
    
    # [임시 루프] 실제 센서 연동 전 테스트용
    while True:
        # data = read_sensor() # 센서값 읽기 로직 (추후 구현)
        # features = extract_features(data) # 10개 특성 추출
        
        # dummy_input = ... (10개 특성)
        # scaled_input = scaler.transform([dummy_input])
        # prob = model(torch.FloatTensor(scaled_input)).item()

        # [C3-2] 80~90% 구간 트리거 (F8-1)
        # if 0.8 <= prob < 0.9:
        #     trigger_validation_mode() # Feature F(YOLO)에게 알림
        
        break # 테스트를 위해 한 번만 돌고 멈춤

if __name__ == "__main__":
    run_inference()