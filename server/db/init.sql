-- =========================================================
-- [PART 1] AI 학습 및 센서 데이터
-- =========================================================

-- 1. 세션 메타데이터 테이블
CREATE TABLE IF NOT EXISTS sensor_sessions (
    session_id      SERIAL PRIMARY KEY,
    predicted_prob  DOUBLE PRECISION,
    yolo_verified   BOOLEAN,
    final_label     INTEGER,
    pm25_slope      DOUBLE PRECISION,
    temp_hum_corr   DOUBLE PRECISION,
    pm_voc_corr     DOUBLE PRECISION,
    pm25_std        DOUBLE PRECISION,
    voc_std         DOUBLE PRECISION,
    pm25_range      DOUBLE PRECISION,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 시계열 상세 로그 테이블
CREATE TABLE IF NOT EXISTS sensor_data_logs (
    session_id      INTEGER REFERENCES sensor_sessions(session_id),
    measured_at     TIMESTAMP NOT NULL,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    pm25            DOUBLE PRECISION,
    voc             DOUBLE PRECISION
);

-- 3. 하이퍼테이블 변환 및 인덱스 (TimescaleDB)
SELECT create_hypertable('sensor_data_logs', 'measured_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_session_id ON sensor_data_logs(session_id);


-- =========================================================
-- [PART 2] 웹앱/로봇 상태 모니터링 (TimescaleDB)
-- =========================================================

CREATE TABLE IF NOT EXISTS robot_status_log (
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    robot_id        VARCHAR(50) NOT NULL,
    
    battery         INTEGER,
    is_charging     BOOLEAN,
    power_status    VARCHAR(20),
    operation_mode  VARCHAR(20),
    movement_status VARCHAR(20),
    
    pm25            DOUBLE PRECISION,
    voc             INTEGER,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    
    pose_x          DOUBLE PRECISION,
    pose_y          DOUBLE PRECISION,
    pose_theta      DOUBLE PRECISION,
    
    air_score       INTEGER,
    air_grade       VARCHAR(20)
);

-- 하이퍼테이블 변환, 인덱스 생성 및 7일 보존 정책 적용
SELECT create_hypertable('robot_status_log', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS robot_status_log_robot_id_time_idx ON robot_status_log (robot_id, time DESC);
SELECT add_retention_policy('robot_status_log', INTERVAL '7 days');


-- =========================================================
-- [PART 3] 로봇 핵심 데이터 (Map, Zone, Dock)
-- =========================================================

-- 1. 맵 데이터 저장 (S3 연동)
CREATE TABLE IF NOT EXISTS robot_maps (
    map_id          SERIAL PRIMARY KEY,
    robot_id        VARCHAR(50) NOT NULL,
    map_name        VARCHAR(100),
    s3_url          TEXT NOT NULL,
    resolution      DOUBLE PRECISION,
    width           INTEGER,
    height          INTEGER,
    origin_x        DOUBLE PRECISION,
    origin_y        DOUBLE PRECISION,
    origin_theta    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_robot_maps_robot_id ON robot_maps(robot_id, created_at DESC);

-- 2. 구역(Zone) 관리 테이블
CREATE TABLE IF NOT EXISTS robot_zones (
    zone_id         SERIAL PRIMARY KEY,
    robot_id        VARCHAR(50) NOT NULL,
    zone_name       VARCHAR(50) NOT NULL,
    center_data     JSONB NOT NULL,
    area_data       JSONB NOT NULL,
    polygon_data    JSONB,
    air_score       INTEGER,
    air_grade       VARCHAR(20),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_robot_zone_name UNIQUE (robot_id, zone_name),
    CONSTRAINT unique_robot_zone_id UNIQUE (robot_id, zone_id)
);

-- 3. 도킹 스테이션 위치 관리
CREATE TABLE IF NOT EXISTS robot_docks (
    robot_id        VARCHAR(50) PRIMARY KEY,
    x               NUMERIC NOT NULL,
    y               NUMERIC NOT NULL,
    theta           NUMERIC NOT NULL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================================================
-- [PART 4] 부가 서비스 (이벤트 로그, 스케줄, QR 토큰)
-- =========================================================

-- 1. 이벤트 로그 테이블
CREATE TABLE IF NOT EXISTS robot_event_logs (
    log_id          SERIAL PRIMARY KEY,
    robot_id        VARCHAR(50) NOT NULL,
    event_type      VARCHAR(20) NOT NULL,
    message         TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. 기상/취침 스케줄 관리
CREATE TABLE IF NOT EXISTS robot_schedules (
    robot_id        VARCHAR(50) PRIMARY KEY,
    wake_time       VARCHAR(5),
    sleep_time      VARCHAR(5),
    is_enabled      BOOLEAN,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 3. QR 인증 토큰 관리
CREATE TABLE IF NOT EXISTS aria_qr_tokens (
    qr_token        VARCHAR(255) PRIMARY KEY,
    robot_id        VARCHAR(50),
    user_name       VARCHAR(50),
    robot_name      VARCHAR(50),
    is_valid        BOOLEAN DEFAULT true
);