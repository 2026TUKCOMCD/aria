import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_session_data(session_id, scenario):
    duration = 1800  # 30분 (초 단위)
    start_time = datetime.now()
    
    # 기본 베이스라인 설정
    t = np.arange(duration)
    temp = 24.0 + np.random.normal(0, 0.1, duration)
    hum = 45.0 + np.random.normal(0, 0.2, duration)
    pm25 = 10.0 + np.random.normal(0, 0.5, duration)
    voc = 0.1 + np.random.normal(0, 0.01, duration)

    # --- 시나리오별 패턴 주입 (핵심 로직) ---
    if scenario == "1.라면_끓이기":
        hum += (t / 60) * 1.5  # 습도 급증
        temp += (t / 60) * 0.3 # 온도 완만 상승
        pm25 += (t / 60) * 1.2 # 미세먼지 소폭 상승
        voc += (t / 60) * 0.15  # 기존 0.05에서 0.15로 상향
    elif scenario == "2.삼겹살_굽기":
        pm25 += (t / 60) * 8.0 # 미세먼지 폭발
        voc += (t / 60) * 0.4  # VOC 동시 상승
        temp += (t / 60) * 0.5 # 강력한 열기
    elif scenario == "3.생선_구이":
        pm25 += (t / 60) * 12.0 # 가장 강력한 미세먼지
        voc += (t / 60) * 0.6
    elif scenario == "4.야채_볶음": 
        pm25 += (t / 60) * 3.0
        voc += (t / 60) * 0.2
    elif scenario == "5.토스트_굽기":
        pm25 += (t / 60) * 1.5
        voc += (t / 60) * 0.1
    elif scenario == "6.가습기_가동":
        hum += (t / 60) * 5.0  # 습도만 폭주
    elif scenario == "7.거실_청소":
        pm25 += 50 * np.exp(-((t-300)**2)/20000) # 일시적 먼지 튐
    elif scenario == "9.향수_분사":
        voc += 2.0 * np.exp(-((t-100)**2)/5000) # VOC만 일시 폭증
    elif scenario == "10.환기":
        pm25 *= 0.5
        hum -= 5.0

    # 데이터프레임 변환 (통계 계산용)
    df = pd.DataFrame({'temp': temp, 'hum': hum, 'pm': pm25, 'voc': voc})
    
    # --- DB 필드값 계산 (sensor_sessions) ---
    pm25_slope = float((pm25[-1] - pm25[0]) / (duration / 60))
    temp_hum_corr = float(df['temp'].corr(df['hum']))
    pm_voc_corr = float(df['pm'].corr(df['voc']))
    
    # NaN 처리 (상관관계 계산 불가 시)
    if np.isnan(temp_hum_corr): temp_hum_corr = 0.0
    if np.isnan(pm_voc_corr): pm_voc_corr = 0.0

    # 세션 메타데이터 (sensor_sessions 테이블 구조)
    session_meta = {
        "session_id": session_id,
        "predicted_prob": 0.0, # AI가 나중에 채울 곳
        "yolo_verified": False,
        "final_label": 1 if "요리" in scenario or session_id <= 5 else 0,
        "pm25_slope": round(pm25_slope, 4),
        "temp_hum_corr": round(temp_hum_corr, 4),
        "pm_voc_corr": round(pm_voc_corr, 4),
        "pm25_std": round(float(np.std(pm25)), 4),
        "voc_std": round(float(np.std(voc)), 4),
        "pm25_range": round(float(np.max(pm25) - np.min(pm25)), 4)
    }

    # 시계열 로그 (sensor_data_logs 테이블 구조)
    logs = []
    for i in range(0, duration, 10): # 10초 단위로 샘플링하여 저장
        logs.append({
            "session_id": session_id,
            "measured_at": (start_time + timedelta(seconds=i)).isoformat(),
            "temperature": round(temp[i], 2),
            "humidity": round(hum[i], 2),
            "pm25": round(pm25[i], 2),
            "voc": round(voc[i], 2)
        })

    return {"meta": session_meta, "logs": logs}

# --- 10가지 시나리오 실행 ---
scenarios = [
    "1.라면_끓이기", "2.삼겹살_굽기", "3.생선_구이", "4.야채_볶음", "5.토스트_굽기",
    "6.가습기_가동", "7.거실_청소", "8.사람_있음", "9.향수_분사", "10.환기"
]

final_output = []
for idx, name in enumerate(scenarios):
    final_output.append(generate_session_data(idx + 1, name))

with open("aria_synthetic_data.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print(f" 총 {len(scenarios)}개의 시나리오 데이터가 'aria_synthetic_data.json'으로 저장되었습니다.")