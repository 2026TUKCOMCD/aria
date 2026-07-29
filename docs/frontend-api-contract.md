# ARIA Web Frontend API Contract

This document lists the REST API and Socket.IO events currently required by the ARIA web frontend.

The local Node server has mock handlers for these APIs. When cloud integration starts, the backend should keep the same URL paths, request bodies, response shapes, and event names as much as possible.

## Base URL

The frontend reads the API server URL from Vite environment variables.

```env
VITE_ARIA_API_URL=http://localhost:3000
VITE_ROBOT_ID=1
VITE_API_SECRET_TOKEN=...
```

For cloud deployment, replace only `VITE_ARIA_API_URL` with the cloud server URL.

## Common Headers

Robot APIs:

```http
Content-Type: application/json
X-ARIA-SECRET: {VITE_API_SECRET_TOKEN}
```

QR auth API:

```http
Authorization: Bearer {qr_token}
X-ARIA-QR-TOKEN: {qr_token}
```

## REST API

### 1. QR Token Verify

```http
GET /auth/verify
```

Checks whether a QR token is valid and returns the robot/user info needed for login.

Response:

```json
{
  "valid": true,
  "robot_id": "1",
  "user_name": "Minjae",
  "robot_name": "ARIA_01"
}
```

If `valid` is `false`, the frontend does not log in and shows an error message.

### 2. Latest Map

```http
GET /robots/{id}/map
```

Returns the latest map image URL and metadata used for coordinate conversion.

Response:

```json
{
  "robot_id": "1",
  "map_name": "local test map",
  "map_url": "https://.../map.png",
  "metadata": {
    "resolution": 0.025,
    "origin": [-2.0, -1.0, 0.0],
    "width": 800,
    "height": 600
  },
  "last_updated": "2026-05-13T12:00:00Z"
}
```

Compatibility:

- Preferred field: `map_url`
- Also accepted by frontend: `image_url`
- If both exist, `map_url` is used first.

Frontend coordinate conversion:

```text
left% = ((x - originX) / (width * resolution)) * 100
top%  = (1 - (y - originY) / (height * resolution)) * 100
```

### 3. Zone List

```http
GET /robots/{id}/zones
```

Returns the room/zone list displayed on the map.

Response:

```json
{
  "robot_id": "1",
  "zones": [
    {
      "id": 1,
      "name": "Living Room",
      "center": { "x": 1.5, "y": 1.2 },
      "area": {
        "x_min": -1.5,
        "y_min": -0.6,
        "x_max": 4.3,
        "y_max": 3.0
      }
    }
  ]
}
```

Notes:

- `center` and `area` must use the same real-world coordinate system as `/robots/{id}/map`.
- If the uploaded map is segmented by rooms, the backend should provide matching zone coordinates.
- If rectangular areas are not enough later, add a `polygon` field.

### 4. Save Zones

```http
PUT /robots/{id}/zones
```

Saves zone names, center coordinates, and optional rectangular areas.

Request:

```json
{
  "zones": [
    {
      "id": 1,
      "name": "Living Room",
      "center": { "x": 1.5, "y": 1.2 },
      "area": {
        "x_min": -1.5,
        "y_min": -0.6,
        "x_max": 4.3,
        "y_max": 3.0
      }
    }
  ]
}
```

Response:

```json
{ "success": true }
```

### 5. Zone Air Quality

```http
GET /robots/{id}/air-quality/zones
```

Returns zone-level air quality data for map overlays.

Response:

```json
{
  "robot_id": "1",
  "update_interval_sec": 30,
  "zones": [
    {
      "zone_id": 1,
      "pm25": 8,
      "voc": 120,
      "status": "GOOD",
      "measured_at": "2026-05-13T12:00:00Z"
    }
  ]
}
```

`status` enum:

```text
GOOD | NORMAL | BAD
```

Frontend stale-data policy:

- If `measured_at` is older than 5 minutes, the frontend displays the zone as `STALE`.
- `STALE` is rendered as a gray overlay.

### 6. Robot Status Summary

```http
GET /robots/{id}/status
```

Returns current robot status and overall air quality score.

Response:

```json
{
  "robot_status": {
    "battery": 82,
    "is_charging": false,
    "power": "OFF",
    "mode": "AUTO",
    "current_zone": "LIVING_ROOM"
  },
  "air_quality": {
    "score": 75,
    "grade": "NORMAL",
    "sensors": {
      "pm25": 25.4,
      "voc": 120,
      "temperature": 24.5,
      "humidity": 45.0
    }
  }
}
```

`power` enum:

```text
ON | OFF | SLEEP
```

`mode` enum:

```text
AUTO | MANUAL | TURBO
```

`grade` enum:

```text
GOOD | NORMAL | BAD | CRITICAL
```

### 7. Robot Command

```http
POST /robots/{id}/command
```

Current frontend command bodies:

```json
{ "target": "MODE", "action": "AUTO" }
```

```json
{ "target": "MODE", "action": "MANUAL" }
```

```json
{ "target": "POWER", "action": "ON" }
```

```json
{ "target": "POWER", "action": "OFF" }
```

```json
{ "target": "SLAM", "action": "ON" }
```

Response:

```json
{ "success": true }
```

### 8. Navigate

```http
POST /robots/{id}/navigate
```

Coordinate navigation:

```json
{ "type": "COORDINATE", "x": 12.5, "y": 5.0 }
```

Zone navigation:

```json
{ "type": "ZONE", "zone_id": 1 }
```

Response:

```json
{ "success": true, "accepted": true }
```

Recommended status code: `202 Accepted`

### 9. Schedule

```http
POST /robots/{id}/schedule
```

Request:

```json
{
  "wake_time": "07:30",
  "sleep_time": "23:00",
  "enabled": true
}
```

Response:

```json
{
  "success": true,
  "message": "Schedule saved."
}
```

Backend behavior:

- Save schedule to cloud DB.
- If `enabled` is true, sync schedule to robot Shadow.
- If `enabled` is false, save schedule but do not auto-control the robot by schedule.

### 10. Reset

```http
POST /robots/{id}/reset
```

Request:

```json
{ "target": "MAP" }
```

or

```json
{ "target": "AI" }
```

Response:

```json
{
  "success": true,
  "message": "Reset command accepted."
}
```

### 11. Event Logs

```http
GET /api/events?robot_id={id}
```

Response:

```json
{
  "success": true,
  "data": [
    {
      "log_id": 1,
      "event_type": "CLEANING",
      "message": "Cooking pollution detected.",
      "created_at": "2026-05-13T12:00:00Z"
    }
  ]
}
```

## Socket.IO Events

The frontend connects to `VITE_ARIA_API_URL` with Socket.IO.

### 1. robot_alert

Server to frontend:

```json
{
  "robot_id": "1",
  "event_type": "NAVIGATE",
  "message": "Navigation command accepted."
}
```

Frontend behavior:

- Adds the event to the event log.
- Shows an alert if `event_type` is `EMERGENCY`.

### 2. status_change

Server to frontend:

```json
"RUNNING"
```

or

```json
"IDLE"
```

Frontend behavior:

- `RUNNING` displays robot operation as active.
- `IDLE` displays robot operation as standby.

### 3. robot_position

Server to frontend:

```json
{
  "robot_id": "1",
  "x": 1.5,
  "y": 2.0,
  "theta": 0.3,
  "updated_at": "2026-05-13T12:00:00Z"
}
```

Coordinate notes:

- `x` and `y` should be real-world coordinates compatible with `/robots/{id}/map` metadata.
- Meter units are recommended.
- `theta` is currently handled as radians by the frontend. If the backend sends degrees, the frontend must be changed.

## Current Mock Server Notes

The local mock APIs are in:

```text
server/cloud/aria_websocket_server/server.js
```

Cloud integration replacement points:

```text
/auth/verify                    -> QR token validation
/robots/:id/map                 -> latest S3 map URL from DB
/robots/:id/zones               -> zone data from DB
/robots/:id/air-quality/zones   -> zone-level air quality from sensors/DB
/robots/:id/status              -> robot Shadow/DB status
/robots/:id/command             -> robot command publishing
/robots/:id/schedule            -> DB save + Shadow sync
/robots/:id/reset               -> reset command publishing
/robots/:id/navigate            -> navigation command publishing
robot_position                  -> MQTT position receive + Socket.IO emit
robot_alert                     -> AI/Lambda event receive + Socket.IO emit
```
