# AWS IoT Core Rule Specification

## Rule: AriaSensorDataToLambda

### 1. Description
로봇(`aria/{id}/data/status`)이 보내는 실시간 상태 메시지를 수집하여, 데이터 구조를 평탄화(Flatten)한 뒤 Lambda 함수로 전달합니다.

### 2. SQL Statement (Updated)
- **Topic Filter**: `aria/+/data/status` (QoS 0)
- **Role**: JSON 내부의 `sensors` 객체를 풀어서 최상위 레벨로 올리고, 토픽에서 로봇 ID를 추출합니다.

```sql
SELECT
    *,                                  -- pose, air_score, air_grade 등 나머지 필드 그대로 전달
    topic(2) as device_id,              -- 토픽의 두 번째 부분(robot_id) 추출
    timestamp() as server_timestamp,    -- 서버 수신 시간(ms)
    mode as operation_mode,             -- DB 컬럼명과 맞춤 (mode -> operation_mode)
    power as power_status,              -- DB 컬럼명과 맞춤 (power -> power_status)
    sensors.temperature as temperature, -- 중첩된 JSON 꺼내기
    sensors.humidity as humidity,
    sensors.pm25 as pm25,
    sensors.voc as voc
FROM 'aria/+/data/status'
```

> **변경 이유 (2026-10 최종 점검)**
> 이전 SQL은 `air_quality.score`, `air_quality.grade`를 읽었지만 로봇(aria_mqtt_node)은
> `air_score`, `air_grade`를 **최상위 필드**로 보낸다. 또 `pose`를 선택하지 않아
> 위치가 항상 0으로 저장됐다. `SELECT *`를 추가해 두 문제를 함께 해결한다.
> AriaSensorSaver는 `pose` 객체와 구버전 `air_quality` 형태를 모두 처리한다.
>
> ⚠️ 이 문서만 바뀌고 실제 규칙은 바뀌지 않는다 — **AWS IoT Core 콘솔 → 메시지 라우팅 → 규칙
> → AriaSensorDataToLambda → SQL 문 편집**에서 위 SQL로 교체해야 한다.