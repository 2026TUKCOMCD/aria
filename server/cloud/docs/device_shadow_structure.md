# AWS IoT Core Device Shadow Structure 정의 (JSON)

# reported(보고된 상태): 로봇이 자신의 상태를 클라우드에 알린 값을 value로 갖는 key(클라우드에게 보고하는 곳)
# desired(희망 상태): 사용자가 웹앱으로 요청한 상태를 value로 갖는 key
# delta(차이): reported와 desired의 차이를 value로 갖는 key

```json
{
  "state": {
    "desired": {
      "power": "OFF", // "ON" | "OFF" 
      "mode": "AUTO", // "AUTO" | "MANUAL" ( Manual: 수동 조작, Auto: 자동(AI 기능 탑재) )
      "speed": 50 //팬 속도 조절 (0~100)
    },
    "reported": {
      "power": "OFF",
      "mode": "AUTO",
      "speed": 50,
      "is_charging": false,      // 충전 중인가?
      "battery": 100
    }
  }
}
```

### 🛡️ Device Shadow 데이터 유효성 검증 규칙 (Validation Rules)

AWS IoT Core Device Shadow를 업데이트할 때, 다음의 데이터 타입과 허용 범위 규칙을 반드시 준수해야 합니다. (추후 API Gateway 및 Lambda 로직에서 이 규칙을 기반으로 에러 처리를 수행합니다.)

- **`power` (제어 가능)**: `String (Enum)` 타입. `"ON"` 또는 `"OFF"` 문자열만 허용.
- **`mode` (제어 가능)**: `String (Enum)` 타입. `"AUTO"`(자율주행 공기청정) 또는 `"MANUAL"`(수동 조작) 문자열만 허용.
- **`speed` (제어 가능)**: `Integer` 타입. `0` ~ `100` 사이의 정수만 허용. (범위 초과 시 400 Bad Request 처리)
- **`is charging` (조회 전용)**: `Boolean` 타입. 로봇의 현재 충전 여부. `true` or `false` 만 허용.
- **`battery` (조회 전용)**: `Integer` 타입. `0` ~ `100` 사이의 정수. (사용자의 `desired` 업데이트 요청에서 제외됨)