-- 1. 세션 메타데이터 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_sessions (
    session_id SERIAL PRIMARY KEY,
    predicted_prob FLOAT,
    yolo_verified BOOLEAN,
    final_label INT,
    pm25_slope FLOAT,
    temp_hum_corr FLOAT,
    pm_voc_corr FLOAT,
    pm25_std FLOAT,
    voc_std FLOAT,
    pm25_range FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 시계열 상세 로그 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_data_logs (
    log_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES sensor_sessions(session_id) ON DELETE CASCADE,
    measured_at TIMESTAMP NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pm25 FLOAT,
    voc FLOAT
);