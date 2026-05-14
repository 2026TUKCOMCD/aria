# AWS IoT Core Rule Specification

## Rule: AriaSensorDataToLambda

### 1. Description
로봇(`aria/{id}/data/status`)이 보내는 실시간 상태 메시지를 수집하여, 데이터 구조를 평탄화(Flatten)한 뒤 Lambda 함수로 전달합니다.

### 2. SQL Statement (Updated)
- **Topic Filter**: `aria/+/data/status` (QoS 0)
- **Role**: JSON 내부의 `sensors` 객체를 풀어서 최상위 레벨로 올리고, 토픽에서 로봇 ID를 추출합니다.

```sql
SELECT 
    topic(2) as device_id,          -- 토픽의 두 번째 부분(robot_id) 추출
    timestamp() as server_timestamp, -- 서버 수신 시간
    battery,
    mode as operation_mode,         -- DB 컬럼명과 맞춤 (mode -> operation_mode)
    power as power_status,          -- DB 컬럼명과 맞춤 (power -> power_status)
    is_charging,
    sensors.temperature as temperature, -- 중첩된 JSON 꺼내기
    sensors.humidity as humidity,
    sensors.pm25 as pm25,
    sensors.voc as voc,
    air_quality.score as air_score,
    air_quality.grade as air_grade
FROM 'aria/+/data/status'