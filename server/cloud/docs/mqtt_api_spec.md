# API & Communication Specification

## 1. HTTP API - Common Standards
> **Base URL**: `https://api.aria-project.com/api/v1` (Example)  
> **Data Format**: JSON (Except for file uploads)  
> **Date Format**: ISO 8601 (e.g., `2026-02-08T12:00:00Z`)

### 1.1 Authentication
All HTTP requests must include the **QR Token** in the Header.
- **Header Key**: `Authorization`
- **Value Format**: `Bearer {QR_TOKEN_STRING}`

### 1.2 HTTP Response Codes
| Code | Status | Description |
|---|---|---|
| **200** | `OK` | Request processed successfully (Read, Update) |
| **201** | `Created` | Resource created successfully (File upload, Data generation) |
| **202** | `Accepted` | Request accepted but processing is pending (Long-running tasks like navigation) |
| **204** | `No Content` | Request successful, but no content to return (Delete) |
| **400** | `Bad Request` | Invalid syntax or missing parameters |
| **401** | `Unauthorized` | Missing or invalid authentication token |
| **403** | `Forbidden` | Valid token but insufficient permissions |
| **404** | `Not Found` | Resource not found |
| **405** | `Method Not Allowed` | HTTP method not supported for this endpoint |
| **500** | `Internal Server Error` | Unexpected server-side error |

### 1.3 Error Response Format (JSON)
When a **4XX** or **5XX** error occurs, the server returns the following JSON structure:

```javascript
{
  "code": "ERROR_CODE_STRING" //ex: QR 토큰 유효성 검사 및 로봇 정보 획득
  "message": "User-friendly error description",
  "timestamp": "2026-02-08T15:30:00Z"
}
```


## 1.4 🌐 HTTP API Endpoints

### 🔐 Auth & Connection
<details>
<summary><code>GET</code> <b>/auth/verify</b> - QR 토큰 유효성 검사 및 로봇 정보 획득</summary>
<br>

- **Description**: 초기 접속 시 QR 토큰을 검증하고 로봇의 기본 정보를 받아옵니다.
- **Request**: Header에 토큰을 포함
- **Recommended Response**: 200 OK
- **Reason**: 유효성 검사 결과를 즉시 반환하므로
- **Response**: 
```json
{
  "valid": true,
  "robot_id": "robot_12345",
  "user_name": "민재",
  "robot_name": "ARIA_01"
}
```

</details>

<details>
<summary><code>GET</code> <b>/robots/{id}/events/stream</b> - 실시간 알림 스트림 (SSE)</summary>
<br>

- **Description**: 웹앱과 연결을 유지하며(Keep-Alive), 이벤트 발생 시 실시간으로 데이터를 푸시(Push) 받습니다.
- **Header**: 
{
  Accept: text/event-stream
  Authorization: Bearer {QR}
}
- **Recommended Response**: 200 OK
- **Reason**: 스트림 연결이 성공했음을 의미 (빠름, delay 없음)
- **Response**:
```text
{
  event: clean_status
  data: {
    "type": "CLEAN_DONE",
    "timestamp": "2026-01-19T14:30:00",
    "message": "거실 청정이 완료되었습니다. 충전 스테이션으로 복귀합니다."
  }
}
```
</details>


### 🤖 Robot Control & Status
<details>
<summary><code>GET</code> <b>/robots/{id}/status</b> - 로봇 실시간 상태 조회</summary>
<br>

- **Description**: 로봇의 배터리, 운전 모드, 현재 구역의 공기질 점수를 조회합니다.
- **Recommended Response**: 200 OK
- **Reason**: 현재 DB에 있는 값을 바로 읽어오므로
- **Response**:
```json
{
  "robot_status": {
    "battery": 82,             // Integer (0-100)
    "is_charging": false,      // Boolean
    "power": "ON",             // Enum: "ON", "OFF", "SLEEP"
    "mode": "AUTO",            // Enum: "AUTO", "MANUAL", "TURBO"
    "current_zone": "LIVING_ROOM" // String (없으면 null)
  },
  "air_quality": {
    "score": 75,               // Integer (0-100, 높을수록 좋음 or 나쁨 기준 정의 필요)
    "grade": "NORMAL",         // Enum: "GOOD", "NORMAL", "BAD", "CRITICAL"
    "sensors": {
      "pm25": 25.4,            // Float (µg/m³)
      "voc": 120,              // Integer (Index or ppb)
      "temperature": 24.5,     // 섭씨 온도 (°C)
      "humidity": 45.0         // 상대 습도 (%)
    }
  }
}
```
</details>

<details>
<summary><code>POST</code> <b>/robots/{id}/command</b> - 운전 모드 제어</summary>
<br>

- **Description**: 로봇의 모드(Auto/Manual)를 변경하거나 동작을 제어합니다.
- **Request Body**: 
{
  "command": "SET_MODE",  // Enum: "POWER", "SET_MODE"
  "value": "TURBO"        // "ON/OFF"
}
- **Recommended Response**: 200 OK
- **Reason**: DB상의 상태(Mode)값만 바꾸는 건 순식간이므로 즉시 성공 처리
- **Response**: 
```json
{"success": true, "message": "Command sent"}
```
</details>

<details>
<summary><code>POST</code> <b>/robots/{id}/reset</b> - 초기화 명령</summary>
<br>

- **Description**: DB 데이터를 삭제하고 로봇을 초기화 상태로 되돌립니다.
- **Request Body**: 
{
  "target": "ALL"  // Enum: "ALL" (전체), "MAP" (지도만)
}
- **Recommended Response**: 202 Accepted
- **Reason**: 로봇이 맵을 지우고 센서를 재설정하는 데 시간이 걸리기 때문 (비동기 처리)
</details>


### 🗺️ Map & Navigation
<details>
<summary><code>GET</code> <b>/robots/{id}/map</b> - 맵 데이터 조회</summary>
<br>

- **Description**: SLAM으로 생성된 최신 맵 이미지 URL과 메타데이터를 조회합니다.
- **Recommended Response**: 200 OK
- **Reason**: 저장된 이미지 URL을 바로 주므로
```json
- **Response**: 
{
  "map_url": "https://s3.ap-northeast-2.amazonaws.com/bucket/map_123.png",
  "metadata": {
    "resolution": 0.05,
    "origin": [-10.5, -5.2, 0.0],
    "width": 800,
    "height": 600
  },
  "last_updated": "2026-01-19T12:00:00Z"
}
```
- **Meaning of tags**:
1. resolution: 0.05
-> 지도 이미지상의 점(Pixel) 1개가 실제 방바닥의 0.05m(5cm) 크기라는 의미
2. width, height
-> 생성된 지도 이미지의 가로 세로 크기[픽섹의 개수]
3. origin의 첫번째, 두번째 인자: 지도 이미지의 왼쪽 아래 구석이 실제 세상의 (0,0) 좌표에서 얼마나 떨어져 있는지를 나타내는 오프셋 값
-> MQTT로 청정기가 보내주는 위치는 (0,0) 기준의 값으로 origin의 값을 더해 로봇의 실제 위치를 파악
4. origin의 마지막 인자
-> Yaw(회전 각도)로서 지도가 얼마나 삐딱하게 기울어져서 그려졌는지를 나타내는 지표
-> 0.0일 때, 지도가 회전하지 않고 똑바로 놓여 있다
</details>

<details>
<summary><code>POST</code> <b>/robots/{id}/map</b> - 맵 데이터 업로드</summary>
<br>

- **Description**: 로봇이 생성한 맵 파일(.pgm)과 메타데이터를 클라우드에 업로드합니다.
- **Header**: 
Header: Authorization: Bearer {QR}
Content-Type: multipart/form-data
- **Request Body**: 
1. map_image: map.png (파일)

2. metadata: JSON 문자열
{
  "resolution": 0.05,
  "origin": [-10.5, -5.2, 0.0],
  "width": 800,
  "height": 600
}
- **Recommended Response**: 201 Created
- **Reason**: 서버(S3)에 새로운 '맵 파일'이라는 자원이 생성되었으므로
</details>

<details>
<summary><code>POST</code> <b>/robots/{id}/navigate</b> - 특정 장소 이동 명령</summary>
<br>

- **Description**: 지도상의 특정 좌표(x, y)로 로봇을 이동시킵니다.
- **Request Body**: 
// Case 1: 좌표 이동
{ "type": "COORDINATE", "x": 12.5, "y": 5.0 }
// Case 2: 방 이동
{ "type": "ZONE", "zone_id": 1 }
- **Recommended Response**: 202 Accepted
- **Reason**: 로봇이 목적지까지 가는 데 수십 초~수 분이 걸리므로 "명령 접수"만 확인
</details>


### 🏠 Zone & Schedule
<details>
<summary><code>GET</code> <b>/robots/{id}/zones</b> - 구역(Room) 목록 조회</summary>
<br>

- **Description**: 설정된 방(Room) 및 금지 구역 목록을 조회합니다.
- **Request Body**: header에 토큰만 있으면 되고, 따로 Body가 필요하지 않음
- **Recommended Response**: 200 OK
- **Reason**: 목록을 바로 보여주므로
- **Response**: 
```json
{
  "robot_id": "robot_123",
  "zones": [
    {
      "id": 1,               // 구역 고유 ID (이동 명령 내릴 때 사용)
      "name": "거실",         // 화면에 표시할 이름
      "center": { "x": 10.5, "y": 5.2 }, // 방의 중심 좌표 (이동 목표 지점)
      "area": {              // (선택) 방의 영역 (사각형)
        "x_min": 5.0, "y_min": 2.0,
        "x_max": 15.0, "y_max": 8.0
      }
    },
    {
      "id": 2,
      "name": "주방",
      "center": { "x": 20.0, "y": 15.0 },
      "area": { ... }
    }
  ]
}
```
</details>

<details>
<summary><code>PUT</code> <b>/robots/{id}/zones</b> - 구역 정보 수정</summary>
<br>

- **Description**: 방 이름이나 구역의 좌표 범위를 수정/등록합니다.
- **Recommended Response**: 200 OK
- **Reason**: 수정한 결과(성공 여부)를 바로 알려주므로
- **Response**: 
```json
{ "success": true }
- ** Request: 
```json
{
  "zones": [
    {
      "id": 1,
      "name": "거실",
      "center": { "x": 10.5, "y": 5.2 }, // 방의 중심 좌표 (이동 목표 지점)
      "area": {              // (선택) 방의 영역 (사각형)
        "x_min": 5.0, "y_min": 2.0,
        "x_max": 15.0, "y_max": 8.0
      }
    },
    { "id": 2, "name": "안방", ... }
  ]
}

```
</details>

<details>
<summary><code>POST</code> <b>/robots/{id}/schedule</b> - 스케줄 설정</summary>
<br>

- **Description**: 기상/취침 시간 및 자동 청소 예약 시간을 설정합니다.
- **Recommended Response**: 200 OK
- **Reason**: 예약 시간을 DB에 저장하는 건 즉시 완료되므로
- **Response**: 
```json
{
  "wake_time": "07:30",  // HH:mm (24시간제)
  "sleep_time": "23:00",
  "enabled": true
}
```
- **부가 설명**:
1. enabled: true
-> 클라우드 DB에 저장된 사용자의 취침 및 기상 스케줄에 맞춰, sleep 혹은 power on 진행
2. enabled: false
-> 클라우드 DB에 저장 후 스케줄에 맞춰서 동작하지는 않음
-> 휴가 시 집 비울 때, 기상 스케줄에 맞춰 로봇 power on되지 않음
</details>

---

## 2. ⚡ MQTT Topics (Real-time)

> **Broker**: AWS IoT Core  
> **Root Topic**: `aria/{id}/...`

### 📤 Cloud → Robot (Commands)
<details>
<summary><b>Control</b> (<code>aria/{id}/cmd/control</code>) - QoS 1</summary>
<br>

- **Direction**: Cloud → Robot
- **Description**: 초기화, 모드 변경, 스케줄 동작 등 핵심 제어 명령
- **Payload**:
```json
{
  "target": "AI_MODE",          // "POWER", "RESET", "AI_MODE"
  "action": "TURN_ON",    // "TURN_ON", "TURN_OFF"
  "timestamp": 1705640000   // Unix Timestamp
}
```
</details>

<details>
<summary><b>Navigation</b> (<code>aria/{id}/cmd/nav</code>) - QoS 1</summary>
<br>

- **Direction**: Cloud → Robot
- **Description**: 위치 이동 및 충전 복귀 명령
- **Payload**: 
```json
{
  "type": "MOVE_TO",
  "x": 12.5,
  "y": 5.0,
  "theta": 0.0 // (선택) 바라보는 방향
}
```
- **Reason(QoS)**: "거실로 가" 명령을 보냈는데 로봇이 못 들으면 안 됨
</details>

<details>
<summary><b>AI Result</b> (<code>aria/{id}/res/predict</code>) - QoS 1</summary>
<br>

- **Direction**: Cloud → Robot
- **Description**: 서버(GPU)에서 분석한 AI 판단 결과 수신
- **Payload**:
```json
 {
  "event_type": "COOKING",   // [ ACTIVITY, INACTIVITY, OUT, COOKING, RETURN ]
  "confidence": 98.5,        // 확률 (%)
  "timestamp": "2026-01-18T19:00:01",
  "action_required": true    // 로봇이 뭔가 해야 하는지 (예: 요리니까 공기청정기 터보 모드)
}
```
- **Reason(QoS)**: 클라우드가 예측한 결과값을 로봇이 못 받으면, 사용자 패턴학습 공기청정기라고 보기 어려움
</details>


### 📥 Robot → Cloud (Data)
<details>
<summary><b>Status Report</b> (<code>aria/{id}/data/status</code>) - QoS 0</summary>
<br>

- **Direction**: Robot → Cloud
- **Description**: 배터리 및 센서 데이터 주기적 보고 (1초 간격)
- **Payload**: 
```json
{
  "battery": 80,        
  "power": "ON",        
  "is_charging": false,
  "mode": "AUTO",
  "pose": { "x": 10.5, "y": 5.2, "theta": 1.57 },
  "sensors": {          // Sensor
    "pm25": 10,
    "voc": 100,
    "temperature": 24.5, 
    "humidity": 45.0      
  },
  "timestamp": "1705640000"
}
```
- **Reason(QoS)**: 
 1~5초마다 계속 보내는 데이터로서, 중간에 하나쯤 빠져도 최신 데이터가 금방 오니까 문제 없음
(네트워크 부하 줄이기)

* 1~5는 임의로 설정한 값이고, 비용적인 측면을 고려해 적절한 주기 상의 필요
</details>

<details>
<summary><b>AI Request</b> (<code>aria/{id}/req/predict</code>) - QoS 1</summary>
<br>

- **Direction**: Robot → Cloud
- **Description**: 엣지 디바이스 트리거 발동 시 AI 정밀 분석 요청
- **Payload**: 
```json
{
  "timestamp": "1705640000",
  "trigger_source": "VOC_SENSOR",  // 무엇 때문에 물어보는지 (디버깅용)
  "sensors": {
    "pir": true,           // 움직임 감지 여부 (활동/비활동)
    "pm25": 80,            // 미세먼지 (요리 감지)
    "voc": 450,            // 냄새/가스 (요리 감지)
    "temperature": 24.5,   // 섭씨 온도 (°C)
    "humidity": 45.0       // 상대 습도 (%)
  }
}
```
- **Reason(QoS)**: 로봇이 "이거 봐주세요" 하고 보낸 건데, 서버가 못 받아서 씹히면 안 됨
</details>

<details>
<summary><b>Data Log</b> (<code>aria/{id}/data/log</code>) - QoS 0</summary>
<br>

- **Direction**: Robot → Cloud
- **Description**: 추후 AI 재학습(Retraining)을 위한 로그 데이터 전송
- **Payload**: 
```json
{
  "request_id": "req_1705640000",   // (선택) 요청 고유 ID
  "timestamp": "2026-01-19T21:30:00", // 트리거 발생 시각 (현재)
  "trigger_source": "VOC_SPIKE",    // 무엇 때문에 보냈는지 (VOC 급증, PIR 미감지 등)
  
  "metadata": {
    "interval_sec": 30,             // 데이터 수집 간격 (30초)
    "total_duration_min": 30,       // 총 데이터 길이 (30분)
    "sample_count": 60              // 배열 안에 들어있는 데이터 개수
  },

  // 핵심: 30분 전부터 현재까지의 데이터 60개를 순서대로 담음
  "sensor_history": [
    {
      "offset_min": -30.0,          // 30분 전
      "pm25": 12.0,
      "voc": 50,
      "temperature": 24.5, 
      "humidity": 45.0,    
      "pir": true
    },
    {
      "offset_min": -29.5,          // 29분 30초 전
      "pm25": 12.5,
      "voc": 55,
      "temperature": 24.5, 
      "humidity": 45.0,
      "pir": true
    },
    // ... (중간 생략) ...
    {
      "offset_min": 0.0,            // 현재 (트리거 발생 시점)
      "pm25": 85.0,   
      "voc": 450,
      "temperature": 24.5, 
      "humidity": 45.0,
      "pir": true
    }
  ]
}
```
- **Reason(QoS)**: 학습 데이터는 대량일 수 있고, 데이터 한두 개 빠져도 AI 성능에 큰 영향 없음. 빠르게 보내는 게 best
</details>

<details>
<summary><b>Event Notification</b> (<code>aria/{id}/event/noti</code>) - QoS 1</summary>
<br>

- **Direction**: Robot → Cloud
- **Description**: 청소 완료, 에러 발생 등 사용자 알림 이벤트 (SSE 중계용)
- **Payload**: 
```json
{
  "type": "CLEAN_DONE", // "BATTERY_LOW", "STUCK", "CLEAN_DONE"
  "message": "청소가 완료되었습니다.",
  "timestamp": 1705640000
}
```
- **Reason(QoS)**: "청소 완료" 알림이 사용자의 앱에 안 뜨면 
오류처럼 보임.
</details>