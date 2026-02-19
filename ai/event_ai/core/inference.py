import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os

# 1. 모델 구조 정의 (성국님의 신경망 구조 유지)
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

# 2. 엔진 초기화
def init_inference_engine(model_path="event_model.pt", scaler_path="scaler.pkl"):
    device = torch.device("cpu")
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("모델 또는 스케일러 파일이 없습니다.")
            return None

        model = AdvancedCookingDetector()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        scaler = joblib.load(scaler_path)
        print("AI Engine: 모델 및 스케일러 로드 완료 (CPU 모드)")
        return {"model": model, "scaler": scaler, "device": device}
    except Exception as e:
        print(f"AI Engine 로드 실패: {e}")
        return None

# 3. 핵심: 10가지 특성 추출 (성국님의 5/15/30분 기울기 로직)
def extract_features(buffer_list):
    """
    900개의 버퍼 데이터를 받아 모델에 넣을 10개의 특성으로 변환합니다.
    """
    df = pd.DataFrame(buffer_list)

    # 현재 시점 데이터 (가장 최신 값)
    current = df.iloc[-1]

    # 기울기 계산 함수
    def get_slope(series, window_size):
        if len(series) < 2: return 0.0
        subset = series.tail(window_size)
        x = np.arange(len(subset))
        y = subset.values
        slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0.0
        return slope

    # 성국님의 10가지 특성 설계
    features = [
        current['pm25'], current['voc'], current['temperature'], current['humidity'],
        get_slope(df['pm25'], 150), get_slope(df['pm25'], 450), get_slope(df['pm25'], 900),
        get_slope(df['voc'], 150), get_slope(df['voc'], 450), get_slope(df['voc'], 900)
    ]

    return np.array(features).reshape(1, -1)

# 4. 실시간 추론 실행
def run_ai_inference(engine, buffer_list):
    if engine is None or len(buffer_list) < 10:
        return {"cooking": 0.0}

    try:
        # 1) 특성 추출
        raw_features = extract_features(buffer_list)

        # 2) 스케일링
        scaled_features = engine['scaler'].transform(raw_features)

        # 3) PyTorch 추론
        input_tensor = torch.FloatTensor(scaled_features).to(engine['device'])
        with torch.no_grad():
            probability = engine['model'](input_tensor).item()

        # 성국님이 원하는 응답 형식 {"cooking": 0.7}
        return {"cooking": float(probability)}

    except Exception as e:
        print(f"⚠️ 추론 중 오류 발생: {e}")
        return {"cooking": 0.0}