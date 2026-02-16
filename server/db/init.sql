-- =========================================================
-- [PART 1] AI 학습용 데이터셋
-- =========================================================

-- 1. 세션 메타데이터 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_sessions (
    session_id      SERIAL PRIMARY KEY,
    predicted_prob  FLOAT,
    yolo_verified   BOOLEAN,
    final_label     INT,
    pm25_slope      FLOAT,
    temp_hum_corr   FLOAT,
    pm_voc_corr     FLOAT,
    pm25_std        FLOAT,
    voc_std         FLOAT,
    pm25_range      FLOAT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 시계열 상세 로그 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_data_logs (
    session_id      INT REFERENCES sensor_sessions(session_id),
    measured_at     TIMESTAMP NOT NULL,
    temperature     FLOAT,
    humidity        FLOAT,
    pm25            FLOAT,
    voc             FLOAT
);

-- 3. 하이퍼테이블 변환 (TimescaleDB 적용)
-- if_not_exists => TRUE 옵션을 넣어야 중복 실행 시 에러가 안 뜸
SELECT create_hypertable('sensor_data_logs', 'measured_at', if_not_exists => TRUE);

-- 4. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_session_id ON sensor_data_logs(session_id);


-- =========================================================
-- [PART 2] 웹앱/로봇 상태 모니터링
-- =========================================================

-- 1. 로봇 상태 로그 테이블 생성
CREATE TABLE IF NOT EXISTS robot_status_log (
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(), -- 시간 (필수)
    robot_id        VARCHAR(50) NOT NULL,               -- 로봇 ID (필수)
    
    -- [Robot Status]
    battery         INTEGER,           -- 배터리 (0-100)
    is_charging     BOOLEAN,           -- 충전 중 여부
    power_status    VARCHAR(20),       -- ON, OFF, SLEEP
    operation_mode  VARCHAR(20),       -- AUTO, MANUAL, TURBO
    current_zone    VARCHAR(50),       -- 거실, 주방 등
    
    -- [Air Quality]
    air_score       INTEGER,           -- 종합 점수
    air_grade       VARCHAR(20),       -- GOOD, BAD 등
    
    -- [Sensors]
    pm25            DOUBLE PRECISION,  -- 미세먼지
    voc             INTEGER,           -- VOC (여기는 INT로 요청됨)
    temperature     DOUBLE PRECISION,  -- 온도
    humidity        DOUBLE PRECISION   -- 습도
);

-- 2. TimescaleDB 하이퍼테이블로 변환
SELECT create_hypertable('robot_status_log', 'time', if_not_exists => TRUE);

-- 3. 인덱스 생성 (로봇 ID + 시간 역순 조회 최적화)
CREATE INDEX IF NOT EXISTS idx_robot_status_log_robot_id_time 
ON robot_status_log (robot_id, time DESC);

-- =========================================================
-- [PART 3] 데이터 보존 정책 (자동 삭제) - 추가 기능
-- =========================================================

-- 웹앱 모니터링 데이터는 영원히 가지고 있을 필요가 없으므로, 1주일(7 days) 지난 데이터는 자동 삭제하여 용량을 관리
SELECT add_retention_policy('robot_status_log', INTERVAL '7 days');

-- =========================================================
-- [PART 4] 맵 데이터 저장 (S3 연동) - 이슈 #131
-- =========================================================

CREATE TABLE IF NOT EXISTS robot_maps (
    map_id          SERIAL PRIMARY KEY,
    robot_id        VARCHAR(50) NOT NULL,
    map_name        VARCHAR(100),       -- 지도 이름 (예: 거실_최종)
    s3_url          TEXT NOT NULL,      -- S3 이미지 주소
    
    -- [Map Metadata]
    resolution      FLOAT,              -- 0.05 (m/pixel)
    width           INTEGER,            -- 이미지 가로 크기
    height          INTEGER,            -- 이미지 세로 크기
    origin_x        FLOAT,              -- 원점 X
    origin_y        FLOAT,              -- 원점 Y
    origin_theta    FLOAT,              -- 원점 회전각
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 로봇별로 최신 지도를 빨리 찾기 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_robot_maps_robot_id ON robot_maps(robot_id, created_at DESC);

-- =========================================================
-- [PART 5] 구역 관리를 위한 테이블 - 이슈#132
-- =========================================================

CREATE TABLE robot_zones (
    zone_id SERIAL PRIMARY KEY,
    robot_id VARCHAR(50) NOT NULL,
    zone_name VARCHAR(50) NOT NULL,
    center_data JSONB NOT NULL,
    area_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 로봇 ID와 방 이름의 조합을 '고유값'으로 묶기 
ALTER TABLE robot_zones ADD CONSTRAINT unique_robot_zone_name UNIQUE (robot_id, zone_name);