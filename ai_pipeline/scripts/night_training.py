import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

SYSTEM_DIR = os.path.expanduser('~/aria_ai_system')
DATA_PATH = os.path.join(SYSTEM_DIR, 'data/aria_daily_log.csv')
MODEL_DIR = os.path.join(SYSTEM_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'aria_rf_model.pkl')

POLLUTION_THRESHOLD = 70.0

def run_night_training():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🌙 심야 자동 학습 파이프라인 시작")

    try:
        from s3_sync_manager import download_from_s3, upload_to_s3
        print("☁️ 오늘 수집한 데이터를 AWS S3에 백업합니다...")
        upload_to_s3()

        print("☁️ AWS S3에서 최신 전체 로그 데이터를 동기화합니다...")
        download_from_s3(DATA_PATH)
    except ImportError:
        print("⚠️ 경고: s3_sync_manager.py 가 아직 없습니다. 로컬 데이터로만 학습을 진행합니다.")

    if not os.path.exists(DATA_PATH):
        print(f"❌ 오류: 학습할 데이터 파일이 없습니다. 경로: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료. 총 {len(df)}개의 누적 레코드가 확인되었습니다.")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    df_polluted = df[df['pm25'] >= POLLUTION_THRESHOLD]

    if len(df_polluted) < 10:
        print(f"⚠️ 오염 발생(PM2.5 >= {POLLUTION_THRESHOLD}) 데이터가 {len(df_polluted)}개로 너무 적습니다. 기존 모델을 유지합니다.")
        return

    print(f"🔍 학습에 유효한 오염 패턴 데이터 {len(df_polluted)}개를 추출했습니다.")

    features = ['hour', 'minute', 'day_of_week']
    target = 'zone_name'

    X = df_polluted[features]
    y = df_polluted[target]

    print("🌲 Random Forest 모델 학습을 시작합니다...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_
    print("\n[📊 AI 패턴 분석 결과: 오염 발생에 영향을 미친 주요 요인]")
    for name, importance in zip(features, importances):
        print(f" - {name}: {importance*100:.1f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 학습 완료! 새 가중치 파일 저장 성공: {MODEL_PATH}")

if __name__ == "__main__":
    run_night_training()
