#!/usr/bin/env python3
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

ENDPOINT     = "adecukeeb0iln-ats.iot.ap-northeast-2.amazonaws.com"
CLIENT_ID    = "1"
THING_NAME   = "aria_robot"

BASE_DIR = Path("/srv/aria/users/hs/ros2_last/src/robot_bringup/robot_bringup")
CERTS_DIR = BASE_DIR / "certs"

PATH_TO_ROOT = str(CERTS_DIR / "AmazonRootCA1.pem")
PATH_TO_CERT = str(CERTS_DIR / "bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-certificate.pem.crt")
PATH_TO_KEY  = str(CERTS_DIR / "bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d-private.pem.key")

# ── [추가] 공기질 점수 계산용 상수 ──
PM25_MAX = 100.0
VOC_MAX = 150.0
THRESHOLD_GOOD = 30.0
THRESHOLD_BAD = 70.0


class AriaMqttNode(Node):
    """AWS IoT Core 전용 통신 노드 (명세서 전체 통합 및 동기화 완성본)."""

    def __init__(self):
        super().__init__("aria_mqtt_node")

        self.current_status = {
            "battery": 100,
            "power": "OFF",
            "is_charging": False,
            "mode": "STANDBY",
            "turbo": "OFF",
            "slam": "OFF",
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "sensors": {"pm25": 0, "voc": 0, "temperature": 0.0, "humidity": 0.0},
            # ── [추가] 실시간 공기질 점수와 등급 상태 ──
            "air_score": 0,
            "air_grade": "좋음"
        }

        # ── ROS2 구독 (Subscribe) ──
        self.create_subscription(String, "/esp32/air_raw", self.air_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.pose_callback, 10)
        self.create_subscription(String, "/aria/robot_event", self._on_robot_event, 10)

        # 컨트롤러 상태 및 이벤트 피드백 구독
        self.create_subscription(String, "/aria/mode_state", self._on_mode_state, 10)
        self.create_subscription(String, "/aria/nav_event", self._on_nav_event, 10)
        
        # 스케줄러 공기질 DB 갱신용 구독
        self.create_subscription(String, "/aria/zone_air_quality", self.zone_air_callback, 10)

        # ── ROS2 발행 (Publish) ──
        self.mode_pub  = self.create_publisher(String, "/aria/mode_command", 10) 
        self.motor_pub = self.create_publisher(String, "/esp32/motor_cmd", 10) 
        self.manual_move_pub = self.create_publisher(String, "/aria/manual_move_command", 10)

        # 주기적 클라우드 상태 보고 전송 (5초)
        self.create_timer(5.0, self.publish_status_to_cloud)

        self._setup_mqtt()
        self.get_logger().info("ARIA MQTT node started")

    def _setup_mqtt(self):
        self.mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        self.mqtt_client.configureEndpoint(ENDPOINT, 8883)
        self.mqtt_client.configureCredentials(PATH_TO_ROOT, PATH_TO_KEY, PATH_TO_CERT)
        self.mqtt_client.configureOfflinePublishQueueing(0)
        self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        self.mqtt_client.configureConnectDisconnectTimeout(10)
        self.mqtt_client.configureMQTTOperationTimeout(5)

        # LWT(유언장) 설정
        lwt_topic = f"aria/{CLIENT_ID}/presence"
        lwt_payload = json.dumps({"status": "Offline", "reason": "Connection Lost"})
        self.mqtt_client.configureLastWill(lwt_topic, lwt_payload, 1)

        # MQTT 접속 및 초기화
        if self.mqtt_client.connect(keepAliveIntervalSecond=60):
            self.get_logger().info("AWS IoT Core 연결 완료 (Keep-Alive 60s, LWT 활성화)")
            
            # 부팅 시 클라우드 섀도우 상태 강제 초기화
            init_shadow = {
                "state": {
                    "reported": {
                        "slam": "OFF",
                        "mode": "STANDBY",
                        "power": "OFF"
                    }
                }
            }
            self.mqtt_client.publish(
                f"$aws/things/{THING_NAME}/shadow/update", 
                json.dumps(init_shadow), 
                1
            )
            self.get_logger().info("디바이스 섀도우 초기 상태(SLAM OFF) 클라우드 강제 동기화 완료")
        else:
            raise RuntimeError("AWS IoT Core 연결 실패")

        # 토픽 구독 설정
        self.mqtt_client.subscribe(f"aria/{CLIENT_ID}/cmd/#", 1, self._cmd_callback)
        self.mqtt_client.subscribe(f"$aws/things/{THING_NAME}/shadow/update/delta", 1, self._shadow_callback)
        self.mqtt_client.subscribe(f"aria/{CLIENT_ID}/res/predict", 1, self._predict_callback)
        
        self.status_topic = f"aria/{CLIENT_ID}/data/status"

    # ──────────────────────────────────────────────
    # ROS2 콜백 및 데이터 갱신
    # ──────────────────────────────────────────────

    def air_callback(self, msg: String):
        """ESP32 데이터 수신 및 실시간 점수(Score)/등급(Grade) 계산"""
        try:
            data = json.loads(msg.data)
            pm25 = float(data.get("pm25", 0.0))
            voc = float(data.get("voc", 0.0))
            
            self.current_status["sensors"]["pm25"]        = pm25
            self.current_status["sensors"]["voc"]         = voc
            self.current_status["sensors"]["temperature"] = data.get("temp", 0.0)
            self.current_status["sensors"]["humidity"]    = data.get("humi", 0.0)

            # ── [추가] 실시간 공기질 점수 계산 로직 ──
            score = max(pm25 / PM25_MAX, voc / VOC_MAX) * 100.0
            score_int = int(min(100.0, max(0.0, score)))

            if score_int >= THRESHOLD_BAD:
                grade = "나쁨"
            elif score_int >= THRESHOLD_GOOD:
                grade = "보통"
            else:
                grade = "좋음"

            self.current_status["air_score"] = score_int
            self.current_status["air_grade"] = grade

        except Exception as e:
            self.get_logger().warn(f"AIR 데이터 파싱 실패: {e}")

    def _on_robot_event(self, msg: String):
        """내부(ROS2) 통합 이벤트를 수신하여 웹앱 실시간 알림으로 릴레이"""
        try:
            payload = json.loads(msg.data)
            
            # 🔥 [수정됨] 노션 명세서에 맞춰 'type' 키를 최우선으로 찾고, 없으면 'event', 둘 다 없으면 'INFO'
            event_type = payload.get("type", payload.get("event", "INFO"))
            message = payload.get("message", "새로운 상태 알림이 있습니다.")

            # 클라우드 발사 함수 호출! 
            self.publish_event_notification(event_type, message)
            
        except Exception as e:
            self.get_logger().error(f"Robot Event 처리 오류: {e}")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw  = math.atan2(siny, cosy)
        self.current_status["pose"]["x"]     = round(pos.x, 3)
        self.current_status["pose"]["y"]     = round(pos.y, 3)
        self.current_status["pose"]["theta"] = round(yaw,   3)

    def zone_air_callback(self, msg: String):
        """스케줄러 노드의 특정 구역 공기질 측정 결과를 클라우드로 릴레이 (DB 갱신용)"""
        try:
            payload = json.loads(msg.data)
            topic = f"aria/{CLIENT_ID}/data/air_quality"

            self.mqtt_client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False),
                1
            )
            self.get_logger().info(
                f"[Cloud DB 전송] Zone: {payload.get('current_zone', 'Unknown')} | "
                f"Score: {payload.get('air_score', 0)} | Grade: {payload.get('air_grade', 'Unknown')}"
            )
        except Exception as e:
            self.get_logger().error(f"공기질 데이터 릴레이 실패: {e}")

    # ──────────────────────────────────────────────
    # 내부(Controller) -> 클라우드 상태 릴레이 콜백
    # ──────────────────────────────────────────────
    def _on_mode_state(self, msg: String):
        """컨트롤러에서 모드가 변경되면 클라우드 상태값 동기화"""
        self.current_status["mode"] = msg.data
        self.get_logger().info(f"[내부 상태 동기화] 컨트롤러 모드 업데이트: {msg.data}")

    def _on_nav_event(self, msg: String):
        """컨트롤러의 내비게이션 이벤트를 웹앱(클라우드)으로 팝업 알림 전송"""
        try:
            payload = json.loads(msg.data)
            event_type = payload.get("event", "NAVIGATION_EVENT")
            target_name = payload.get("zone_name", "목적지")

            if event_type == "NAVIGATION_SUCCEEDED":
                message = f"{target_name}에 도착했습니다."
            elif event_type == "NAVIGATION_FAILED":
                message = f"{target_name} 이동에 실패했습니다."
            else:
                message = f"이동 상태 알림: {event_type}"

            self.publish_event_notification(event_type, message)
        except Exception as e:
            self.get_logger().error(f"Nav Event 처리 오류: {e}")

    # ──────────────────────────────────────────────
    # 클라우드 발행 (Publish) 메서드 모음
    # ──────────────────────────────────────────────

    def publish_status_to_cloud(self):
        """[명세 반영] 주기적 상태 보고 (data/status) - 점수와 등급 포함됨"""
        self.current_status["timestamp"] = str(int(time.time()))
        try:
            self.mqtt_client.publish(
                self.status_topic,
                json.dumps(self.current_status),
                0 
            )
            self.get_logger().info(
                f"[Cloud 상태보고] PM2.5={self.current_status['sensors']['pm25']}, "
                f"Score={self.current_status['air_score']} ({self.current_status['air_grade']})"
            )
        except Exception as e:
            self.get_logger().warn(f"상태 보고 전송 실패: {e}")

    def publish_event_notification(self, event_type: str, message: str):
        topic = f"aria/{CLIENT_ID}/event/noti"
        payload = {
            "type": event_type,
            "message": message,
            "timestamp": str(int(time.time()))
        }
        try:
            self.mqtt_client.publish(topic, json.dumps(payload), 1)
            self.get_logger().info(f"[이벤트 알림 발행] {event_type}: {message}")
        except Exception as e:
            self.get_logger().error(f"이벤트 알림 발행 실패: {e}")

    def publish_predict_request(self, trigger_source: str = "VOC_SENSOR", pir_status: bool = True):
        topic = f"aria/{CLIENT_ID}/req/predict"
        payload = {
            "timestamp": str(int(time.time())),
            "trigger_source": trigger_source,
            "sensors": {
                "pir": pir_status,
                "pm25": self.current_status["sensors"]["pm25"],
                "voc": self.current_status["sensors"]["voc"],
                "temperature": self.current_status["sensors"]["temperature"],
                "humidity": self.current_status["sensors"]["humidity"]
            }
        }
        try:
            self.mqtt_client.publish(topic, json.dumps(payload), 1)
            self.get_logger().info(f"[AI 예측 요청 발행] Trigger: {trigger_source}")
        except Exception as e:
            self.get_logger().error(f"AI 예측 요청 발행 실패: {e}")

    # ──────────────────────────────────────────────
    # MQTT 콜백 (클라우드 → 로봇 수신)
    # ──────────────────────────────────────────────

    def _cmd_callback(self, client, userdata, message):
        """[명세 반영] 제어 및 이동 명령 수신 (cmd/#)"""
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            topic = message.topic
            self.get_logger().info(f"[MQTT 수신] topic={topic}, payload={payload}")
            
            # 1. 기존 target/action 방식 명령 처리
            if "target" in payload and "action" in payload:
                target = payload.get("target")
                action = payload.get("action")
                self.get_logger().info(f"[제어 명령 수신] Target: {target}, Action: {action}")
                
                if target == "POWER":
                    if action == "TURN_OFF":
                        stop = String()
                        stop.data = json.dumps({"left": 0, "right": 0, "mode": 0})
                        self.motor_pub.publish(stop)
                        self.current_status["power"] = "OFF"
                        self.get_logger().info("명령 수행: 시스템 전원 정지")
                    elif action == "TURN_ON":
                        self.current_status["power"] = "ON"
                        self.get_logger().info("명령 수행: 시스템 전원 가동")

                elif target == "AI_MODE":
                    if action == "TURN_ON":
                        self.current_status["mode"] = "AI_MODE"
                        mode_msg = String()
                        mode_msg.data = "AI_MODE"
                        self.mode_pub.publish(mode_msg)
                        self.get_logger().info("명령 수행: AI 모드 활성화")
                    elif action == "TURN_OFF":
                        self.current_status["mode"] = "STANDBY"
                        mode_msg = String()
                        mode_msg.data = "STANDBY"
                        self.mode_pub.publish(mode_msg)
                        self.get_logger().info("명령 수행: AI 모드 비활성화")

            # 2. 내비게이션 좌표 이동 명령 처리
            elif payload.get("type") == "MOVE_TO" or ("x" in payload and "y" in payload):
                target_x = float(payload["x"])
                target_y = float(payload["y"])
                target_theta = float(payload.get("theta", 0.0))

                self.get_logger().info(
                    f"[Nav 명령 수신] 목적지: X={target_x}, Y={target_y}, Theta={target_theta}"
                )

                move_payload = {
                    "type": "MOVE_TO",
                    "x": target_x,
                    "y": target_y,
                    "theta": target_theta
                }
                move_msg = String()
                move_msg.data = json.dumps(move_payload, ensure_ascii=False)
                self.manual_move_pub.publish(move_msg)

        except Exception as e:
            self.get_logger().error(f"명령 처리 오류: {e}")

    def _predict_callback(self, client, userdata, message):
        """[명세 반영] AI 판단 결과 수신 (res/predict)"""
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            event_type = payload.get("event_type", "UNKNOWN")
            confidence = payload.get("confidence", 0.0)
            action_req = payload.get("action_required", False)
            
            self.get_logger().info(
                f"[AI 결과 수신] 이벤트: {event_type} | 신뢰도: {confidence}% | 후속조치 필요: {action_req}"
            )
            
            if action_req:
                self.get_logger().info("-> AI 판단에 따른 즉각적인 대응 시스템을 가동합니다.")
                
        except Exception as e:
            self.get_logger().error(f"AI 결과 처리 오류: {e}")

    def _shadow_callback(self, client, userdata, message):
        """웹 대시보드 디바이스 섀도우 변경 동기화 처리"""
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            delta   = payload.get("state", {})
            self.get_logger().info(f"[Shadow Delta 수신] {delta}")

            # 1. 모드 (mode) 변경 처리
            if "mode" in delta:
                new_mode = delta["mode"] 
                self.current_status["mode"] = new_mode
                
                if new_mode == "AUTO":
                    ctrl_mode = "AI"
                elif new_mode == "MANUAL":
                    ctrl_mode = "BASIC"
                elif new_mode == "WAIT":
                    ctrl_mode = "STANDBY"
                else:
                    ctrl_mode = "STANDBY"

                mode_msg = String()
                mode_msg.data = json.dumps({"mode": ctrl_mode})
                self.mode_pub.publish(mode_msg)
                self.get_logger().info(f"[Shadow 동기화] 모드 변경 요청 발행: {ctrl_mode}")

            # 2. SLAM (slam) 변경 처리
            if "slam" in delta:
                new_slam = delta["slam"]
                self.current_status["slam"] = new_slam
                
                if new_slam == "ON":
                    mapping_msg = String()
                    mapping_msg.data = json.dumps({"mode": "MAPPING"})
                    self.mode_pub.publish(mapping_msg)
                    self.get_logger().info("[Shadow 동기화] 매핑(SLAM) 모드 진입 요청")
                else:
                    standby_msg = String()
                    standby_msg.data = json.dumps({"mode": "STANDBY"})
                    self.mode_pub.publish(standby_msg)
                    self.get_logger().info("[Shadow 동기화] 매핑 종료, 대기 모드 복귀")

            # 3. 전원 (power) 변경 처리
            if "power" in delta:
                new_power = delta["power"]
                self.current_status["power"] = new_power
                
                if new_power == "OFF":
                    stop_msg = String()
                    stop_msg.data = json.dumps({"left": 0, "right": 0, "mode": 0})
                    self.motor_pub.publish(stop_msg)
                    
                    standby_msg = String()
                    standby_msg.data = json.dumps({"mode": "STANDBY"})
                    self.mode_pub.publish(standby_msg)
                    self.get_logger().info("[Shadow 동기화] 시스템 전원 OFF (정지 및 대기 전환)")

            # 4. 터보 (turbo) 변경 처리
            if "turbo" in delta:
                new_turbo = delta["turbo"]
                self.current_status["turbo"] = new_turbo
                self.get_logger().info(f"[Shadow 동기화] 터보 모드 변경: {new_turbo}")

            # 델타(명령) 처리가 끝난 후 클라우드에 "반영 완료(reported)" 보고
            reported_payload = {
                "state": {
                    "reported": delta 
                }
            }
            self.mqtt_client.publish(
                f"$aws/things/{THING_NAME}/shadow/update",
                json.dumps(reported_payload),
                1
            )
            self.get_logger().info(f"[Shadow Delta 해소] Reported 상태 클라우드 갱신 완료: {delta}")

        except Exception as e:
            self.get_logger().error(f"Shadow 처리 오류: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AriaMqttNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
