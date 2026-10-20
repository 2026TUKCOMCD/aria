"""ARIA 심야 재학습 파이프라인.

[학습 문제 정의 — 이전 버전과의 차이]
  이전: 오염 행(pm25 >= 70)만 골라 '어느 구역인가(zone_name)'를 맞히는 다중분류.
        → "오염이 났다면 어느 구역일까"를 배우므로, 어떤 시각을 넣어도
          반드시 어떤 구역 하나를 높은 확률로 답한다(오염이 없는 시간대에도).
  현재: 전체 순회 기록으로 '이 구역이 이 시각에 오염되는가(0/1)'를 맞히는 이진분류.
        입력  = 시각(주기 인코딩) + 요일(주기 인코딩) + 구역(one-hot)
        정답  = polluted (pm25 >= POLLUTION_THRESHOLD)
        → 예측 시 구역마다 P(오염)을 따로 구하므로, 70%는 진짜 '그 구역이
          오염될 확률'이 되고, 아무 구역도 넘지 않으면 출동하지 않는다.

[저장 형식]
  joblib 번들(dict): model, feature_columns, zones, threshold, metrics, trained_at
  ai_cleaner_node는 이 번들만 받아들인다(구버전 단일 모델 파일은 무시).
"""
import os
import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

SYSTEM_DIR = os.path.expanduser('~/aria_ai_system')
DATA_PATH = os.path.join(SYSTEM_DIR, 'data/aria_daily_log.csv')
MODEL_DIR = os.path.join(SYSTEM_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'aria_rf_model.pkl')

POLLUTION_THRESHOLD = 70.0   # PM2.5 µg/m³, 이 이상이면 '오염'
MIN_POSITIVE = 10            # 오염 사례가 이보다 적으면 학습하지 않음(기존 모델 유지)
MIN_NEGATIVE = 10            # 비오염 사례도 최소한은 있어야 경계를 배울 수 있음
HOLDOUT_RATIO = 0.2          # 평가용으로 떼어둘 '가장 최근' 데이터 비율

BUNDLE_VERSION = 2


def build_features(df: pd.DataFrame, zones=None) -> pd.DataFrame:
    """학습·추론 공용 피처 생성. ai_cleaner_node도 동일한 규칙을 써야 한다.

    - 시각은 23시와 0시가 가깝다는 걸 모델이 알 수 있게 sin/cos로 인코딩한다.
    - 요일도 일요일(6)과 월요일(0)이 이어지도록 sin/cos로 인코딩한다.
    - 구역은 one-hot. zones를 주면 그 목록 기준으로 컬럼을 고정한다.
    """
    minutes = df['hour'].astype(float) * 60.0 + df['minute'].astype(float)
    dow = df['day_of_week'].astype(float)

    feats = pd.DataFrame({
        'tod_sin': np.sin(2 * math.pi * minutes / 1440.0),
        'tod_cos': np.cos(2 * math.pi * minutes / 1440.0),
        'dow_sin': np.sin(2 * math.pi * dow / 7.0),
        'dow_cos': np.cos(2 * math.pi * dow / 7.0),
    }, index=df.index)

    zone_series = df['zone_name'].astype(str)
    if zones is None:
        zones = sorted(zone_series.unique())
    for z in zones:
        feats[f'zone={z}'] = (zone_series == z).astype(int)
    return feats


def sync_with_s3():
    """로컬 → S3 업로드가 성공했을 때만 S3 → 로컬로 내려받는다.

    업로드가 실패했는데 내려받으면 S3의 옛 파일이 로컬 최신 데이터를 덮어써
    오늘 수집분이 사라진다.
    """
    try:
        from s3_sync_manager import download_from_s3, upload_to_s3
    except ImportError:
        print("⚠️ s3_sync_manager.py 가 없습니다. 로컬 데이터로만 학습합니다.")
        return

    print("☁️ 오늘 수집한 데이터를 AWS S3에 백업합니다...")
    if upload_to_s3():
        print("☁️ AWS S3에서 최신 전체 로그 데이터를 동기화합니다...")
        download_from_s3(DATA_PATH)
    else:
        print("⚠️ S3 업로드 실패 — 로컬 데이터 보호를 위해 다운로드를 건너뜁니다.")


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    before = len(df)
    df = df.dropna(subset=['hour', 'minute', 'day_of_week', 'zone_name', 'pm25'])
    df = df[df['pm25'] >= 0]  # 센서 미준비 값(-1) 제거
    print(f"✅ 데이터 로드: {before}행 → 유효 {len(df)}행")
    return df.reset_index(drop=True)


def evaluate(df: pd.DataFrame, zones) -> dict:
    """시간 순서상 마지막 HOLDOUT_RATIO를 떼어 평가한다.

    무작위 분할을 쓰면 같은 날의 앞뒤 기록이 학습·평가에 섞여 성능이 부풀려진다.
    정확도(accuracy)는 '항상 비오염'이라고만 답해도 높게 나오므로 보고하지 않는다.
    """
    n_test = int(len(df) * HOLDOUT_RATIO)
    if n_test < 5:
        return {'note': 'holdout too small'}

    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    if train['polluted'].nunique() < 2 or test['polluted'].sum() == 0:
        return {'note': 'holdout has no positive samples'}

    m = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=3,
                               class_weight='balanced', random_state=42)
    m.fit(build_features(train, zones), train['polluted'])
    pred = m.predict(build_features(test, zones))
    return {
        'holdout_rows': int(n_test),
        'holdout_positive': int(test['polluted'].sum()),
        'recall': float(recall_score(test['polluted'], pred, zero_division=0)),
        'precision': float(precision_score(test['polluted'], pred, zero_division=0)),
    }


def run_night_training():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 🌙 심야 자동 학습 파이프라인 시작")
    sync_with_s3()

    if not os.path.exists(DATA_PATH):
        print(f"❌ 학습할 데이터 파일이 없습니다: {DATA_PATH}")
        return

    try:
        df = load_clean_data()
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 핵심 변경: 오염 행만 거르지 않고, 전체 행에 오염 여부 라벨을 붙인다.
    df['polluted'] = (df['pm25'] >= POLLUTION_THRESHOLD).astype(int)
    n_pos = int(df['polluted'].sum())
    n_neg = int(len(df) - n_pos)
    print(f"🔍 오염 {n_pos}건 / 비오염 {n_neg}건")

    if n_pos < MIN_POSITIVE or n_neg < MIN_NEGATIVE:
        print(f"⚠️ 오염 {MIN_POSITIVE}건·비오염 {MIN_NEGATIVE}건 이상이 필요합니다. 기존 모델을 유지합니다.")
        return

    zones = sorted(df['zone_name'].astype(str).unique())
    metrics = evaluate(df, zones)
    print(f"📏 시간순 홀드아웃 평가: {metrics}")

    # 데이터가 적으므로 깊이를 줄이고 잎 최소 샘플을 둬서 과적합을 억제한다.
    model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=3,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    X = build_features(df, zones)
    model.fit(X, df['polluted'])

    bundle = {
        'version': BUNDLE_VERSION,
        'model': model,
        'feature_columns': list(X.columns),
        'zones': zones,
        'pollution_threshold': POLLUTION_THRESHOLD,
        'metrics': metrics,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
    }

    # 원자적 교체: 추론 노드가 쓰다 만 파일을 읽지 않도록 임시 파일에 쓴 뒤 rename.
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_path = MODEL_PATH + '.tmp'
    joblib.dump(bundle, tmp_path)
    os.replace(tmp_path, MODEL_PATH)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 🎉 학습 완료: {MODEL_PATH}")


if __name__ == "__main__":
    run_night_training()
