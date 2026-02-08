# shared/constants.py

class MqttTopic:
    """
    [A1-1] MQTT Topic Definitions based on Spec
    Usage: MqttTopic.CMD_CONTROL.format(id='robot_01')
    """
    
    # Root
    TOPIC_ROOT = "aria"

    # --- 1. Cloud -> Robot (Server Publishes, Robot Subscribes) ---
    # 제어 및 초기화: 초기화, 모드 변경, 스케줄 동작 등
    CMD_CONTROL = "aria/{id}/cmd/control"
    
    # 이동 명령: 위치 이동 및 충전 복귀
    CMD_NAV = "aria/{id}/cmd/nav"
    
    # AI 결과: AI 판단 결과 송신
    RES_PREDICT = "aria/{id}/res/predict"

    # --- 2. Robot -> Cloud (Robot Publishes, Server Subscribes) ---
    # 상태 보고: 배터리 + 센서 데이터 주기적 보고
    DATA_STATUS = "aria/{id}/data/status"
    
    # AI 예측 요청: 예측 트리거값 감지 시 요청
    REQ_PREDICT = "aria/{id}/req/predict"
    
    # 학습 데이터: 재학습용 로그 데이터
    DATA_LOG = "aria/{id}/data/log"
    
    # 이벤트 알림: 청정 완료 알림 (SSE 중계용)
    EVENT_NOTI = "aria/{id}/event/noti"


class HttpEndpoint:
    """
    [A4-1, A5-2] HTTP API Endpoints
    Description: Define API routes for Frontend <-> Server communication
    Usage: f"{SERVER_URL}{HttpEndpoint.ROBOT_STATUS.format(id='robot_01')}"
    """
    
    # --- 1. Authentication & Setup ---
    # QR 토큰 검증: 초기 접속 시 로봇 정보 획득 및 유효성 검사 (GET)
    AUTH_VERIFY = "/auth/verify"

    # --- 2. Robot Control (Command) ---
    # 로봇 초기화: DB 데이터 삭제 및 공장 초기화 명령 (POST)
    ROBOT_RESET = "/robots/{id}/reset"
    
    # 운전 모드 제어: Auto/Manual 모드 변경 및 동작 제어 (POST)
    ROBOT_CMD = "/robots/{id}/command"
    
    # 상태 조회: 로봇의 실시간 상태 및 공기질 점수 조회 (GET)
    ROBOT_STATUS = "/robots/{id}/status"
    
    # --- 3. Map Management (S3 Linked) ---
    # 맵 데이터 조회: 최신 맵 이미지 URL 및 메타데이터 조회 (GET)
    MAP_GET = "/robots/{id}/map"
    
    # 맵 데이터 업로드: SLAM으로 생성된 맵 파일 업로드 (POST)
    MAP_UPLOAD = "/robots/{id}/map"
    
    # --- 4. Zone & Navigation ---
    # 구역 목록 조회: 설정된 방(Zone) 좌표 목록 조회 (GET)
    ZONES_GET = "/robots/{id}/zones"
    
    # 구역 수정: 방 이름 및 좌표 수정/등록 (PUT)
    ZONES_UPDATE = "/robots/{id}/zones"
    
    # 특정 장소 이동: 지정된 좌표로 로봇 이동 명령 (POST)
    NAVIGATE = "/robots/{id}/navigate"
    
    # --- 5. Schedule & Event ---
    # 스케줄 설정: 기상/취침 및 청소 예약 시간 설정 (POST)
    SCHEDULE = "/robots/{id}/schedule"
    
    # 실시간 알림 스트림: SSE(Server-Sent Events) 연결 엔드포인트 (GET)
    EVENT_STREAM = "/robots/{id}/events/stream"


# --- QoS Levels ---
# 메시지 전송 품질 설정 (0: 최대 1회, 1: 최소 1회)
QOS_AT_MOST_ONCE = 0
QOS_AT_LEAST_ONCE = 1