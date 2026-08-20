import json
import os
import sys
import time
import threading
from datetime import datetime

PROJECT_ROOT = os.environ.get("ARIA_PROJECT_ROOT", "/home/hs/aria")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai"))

import requests
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# torch 등 무거운 AI 의존성이 없는 환경(예: GPU 없는 라즈베리파이)에서는
# 이 노드 자체가 아무 것도 하지 않고 대기만 한다 (aria_controller_node 등
# 다른 노드는 이 노드의 생사와 무관하게 계속 동작한다 — 이게 이 노드를
# aria_controller_node에서 분리해낸 이유다).
try:
    from ai.event_ai.core.inference import run_ai_inference, init_inference_engine
    from ai.event_ai.core.buffer import AirQualityBuffer
    from activity_ai.smart_activity import SmartActivityDetector
    from activity_ai.camera_recorder import CameraRecorder
    from activity_ai.data_manager import DataManager
    from activity_ai.power_manager import PowerManager
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI 모듈 임포트 실패 (요리감지 파이프라인 비활성화): {e}")
    print(f"ARIA_PROJECT_ROOT={PROJECT_ROOT} 를 확인하세요.")
    AI_MODULES_AVAILABLE = False


class AriaCookingAiNode(Node):
    """ARIA 요리감지 AI 노드.

    aria_controller_node에서 분리되었다 — torch/YOLO 등 무거운 의존성을
    가진 이 노드가 죽거나 느려져도 로봇의 이동/순회/모드전환은 영향받지
    않아야 하기 때문이다.

    구독 토픽:
        /esp32/air_raw  → 공기질 원본 데이터 (버퍼 적재용)
        /aria/ai_active → aria_controller_node가 AI모드에서 순회를 끝냈을 때만
                           true를 보낸다. true일 때만 이 노드가 실제로 동작한다
                           (순회 재실행 중에는 반드시 false).

    동작 (/aria/ai_active == true 인 동안):
        공기질 급변/고농도 감지 → AI 추론 → Nav2로 주방 이동 → 녹화 → YOLO → 클라우드
    """

    def __init__(self):
        super().__init__("aria_cooking_ai_node")

        # 파라미터
        self.declare_parameter(
            "model_path",
            os.path.join(PROJECT_ROOT, "ai/event_ai/models/event_model.pt")
        )
        self.declare_parameter(
            "scaler_path",
            os.path.join(PROJECT_ROOT, "ai/event_ai/models/scaler.pkl")
        )
        self.declare_parameter("video_output_dir", "/home/jj/aria/data/videos")
        self.declare_parameter("kitchen_zone_name", "kitchen")  # 서버 zone 이름 기준

        model_path  = self.get_parameter("model_path").value
        scaler_path = self.get_parameter("scaler_path").value
        video_dir   = self.get_parameter("video_output_dir").value
        self.kitchen_zone_name = self.get_parameter("kitchen_zone_name").value

        # 클라우드 설정
        self.CLOUD_URL    = os.getenv("ARIA_LOG_API_URL")
        self.SECRET_TOKEN = os.getenv("ARIA_SECRET_TOKEN")
        self.ROBOT_ID     = os.getenv("ROBOT_ID", "1")
        self.SERVER_URL   = os.getenv("ARIA_API_URL")

        # 상태
        self._ai_active    = False  # aria_controller_node가 /aria/ai_active로 제어
        self.is_analyzing  = False
        self.last_event_time = 0

        # 임계값 (레거시 프로토타입 및 aria_main_node와 동일한 값 유지)
        self.PM25_SLOPE_THRESHOLD = 0.5
        self.VOC_SLOPE_THRESHOLD  = 0.3
        self.PM25_HIGH_THRESHOLD  = 100.0
        self.VOC_HIGH_THRESHOLD   = 150.0
        self.PROB_THRESHOLD       = 0.70
        self.COOLDOWN_SECONDS     = 60

        # 360도 회전 시간
        self.KITCHEN_RECORD_SEC = 29.0

        # AI 모듈 초기화 (torch 등이 없으면 요리감지 파이프라인만 비활성화)
        self.ai_enabled = AI_MODULES_AVAILABLE
        if self.ai_enabled:
            self.engine        = init_inference_engine(model_path=model_path, scaler_path=scaler_path)
            self.vision_module = SmartActivityDetector()
            self.recorder      = CameraRecorder(output_dir=video_dir, resolution=(640, 480))
            self.dm            = DataManager()
            self.power_manager = PowerManager(sleep_time="23:00", wake_time="07:00", verbose=False)
            self.aq_buffer     = AirQualityBuffer(max_len=900)
        else:
            self.engine = self.vision_module = self.recorder = None
            self.dm = self.power_manager = self.aq_buffer = None

        # 로컬 백업 폴더
        self.backup_dir = os.path.join(PROJECT_ROOT, "ai/event_ai/core/failed_logs")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Nav2 액션 클라이언트 (주방 이동 전용 — aria_controller_node/scheduler와는
        # /aria/ai_active를 통해 활성 구간이 겹치지 않도록 상태머신이 보장한다)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ROS2 구독
        self.create_subscription(String, "/esp32/air_raw", self.air_callback, 10)
        self.create_subscription(Bool,   "/aria/ai_active", self.ai_active_callback, 10)

        # ROS2 발행
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 2초 주기 AI 트리거 체크
        self.create_timer(2.0, self.ai_timer_callback)

        self.get_logger().info(f"ARIA Cooking AI Node 시작 [Robot ID: {self.ROBOT_ID}]")

    # ──────────────────────────────────────────────
    # ROS2 콜백
    # ──────────────────────────────────────────────

    def air_callback(self, msg: String):
        if self.aq_buffer is None:  # 요리감지 AI 비활성화 상태 (torch 등 미설치)
            return
        try:
            data = json.loads(msg.data)
            self.aq_buffer.add_data(
                temp=data.get("temp", 0.0),
                humi=data.get("humi", 0.0),
                pm25=data.get("pm25", 0.0),
                voc=data.get("voc",  0.0)
            )
        except Exception as e:
            self.get_logger().warn(f"AIR 데이터 파싱 실패: {e}")

    def ai_active_callback(self, msg: Bool):
        self._ai_active = bool(msg.data)
        if not self._ai_active:
            self.get_logger().info("⏸️ aria_controller_node 요청으로 요리감지 대기 상태로 전환합니다.")

    # ──────────────────────────────────────────────
    # AI 트리거 체크 (2초 타이머)
    # ──────────────────────────────────────────────

    def ai_timer_callback(self):
        if not self.ai_enabled:  # torch 등 AI 모듈 미설치 -> 요리감지 파이프라인 스킵
            return
        if not self._ai_active:  # aria_controller_node가 순회 중이거나 AI모드가 아님
            return
        if self.is_analyzing:
            return
        if len(self.aq_buffer.buffer) < 150:
            return
        if not self.power_manager.should_process_activity():
            return

        features = self.aq_buffer.get_session_features()
        if not features:
            return

        is_pm_slope  = features.get("pm25_slope",    0) > self.PM25_SLOPE_THRESHOLD
        is_voc_slope = features.get("voc_slope",     0) > self.VOC_SLOPE_THRESHOLD
        is_pm_high   = features.get("current_pm25",  0) > self.PM25_HIGH_THRESHOLD
        is_voc_high  = features.get("current_voc",   0) > self.VOC_HIGH_THRESHOLD

        if (is_pm_slope or is_voc_slope or is_pm_high or is_voc_high):
            if time.time() - self.last_event_time >= self.COOLDOWN_SECONDS:
                t = threading.Thread(target=self._run_analysis_pipeline, daemon=True)
                t.start()

    # ──────────────────────────────────────────────
    # AI 분석 파이프라인 (별도 스레드)
    # ──────────────────────────────────────────────

    def _run_analysis_pipeline(self):
        self.is_analyzing = True
        try:
            # 1단계: AI 추론
            prob_res = run_ai_inference(self.engine, self.aq_buffer.buffer)
            prob = prob_res.get("cooking", 0.0)
            self.get_logger().info(f"[AI] 요리 확률: {prob*100:.1f}%")

            if prob < self.PROB_THRESHOLD:
                return

            # 2단계: 주방 좌표 가져오기
            kitchen_pos = self._get_kitchen_position()
            if kitchen_pos is None:
                self.get_logger().error("[AI] 주방 좌표를 찾을 수 없습니다. 파이프라인 중단.")
                return

            # 3단계: 주방으로 이동 + 동시에 경로 녹화
            self.get_logger().info("[이동] 주방으로 Nav2 이동 시작 + 경로 녹화")
            self.recorder.start_recording(mode="corridor")
            arrived = self._navigate_to(kitchen_pos["x"], kitchen_pos["y"], timeout=60.0)
            corridor_path = self.recorder.stop_recording()

            if not arrived:
                self.get_logger().warn("[이동] 주방 도착 실패 또는 타임아웃")
                return

            time.sleep(1.0)

            # 4단계: 360도 스캔 + 녹화
            self.get_logger().info(f"[Camera] 360도 녹화 중 ({self.KITCHEN_RECORD_SEC}초)")
            self.recorder.start_recording(mode="360")
            self._rotate_360(duration_sec=self.KITCHEN_RECORD_SEC)
            kitchen_path = self.recorder.stop_recording()

            # 5단계: YOLO 검증
            self.get_logger().info("[YOLO] 영상 분석 중...")
            yolo_res = self.vision_module.detect_cooking_event(
                corridor_video=corridor_path,
                kitchen_video=kitchen_path
            )
            self.get_logger().info(
                f"[YOLO] {yolo_res['reason']} (확정: {yolo_res['confirmed']})"
            )

            # 6단계: 클라우드 전송
            features = self.aq_buffer.get_session_features()
            final_package = self.aq_buffer.make_package(
                self.ROBOT_ID, prob, yolo_res["confirmed"], features
            )
            self._upload_to_cloud(final_package)

            # 7단계: 영상 메타데이터 저장
            self.dm.save_cooking_event(
                result=yolo_res,
                corridor_video=corridor_path,
                kitchen_video=kitchen_path
            )

            self.last_event_time = time.time()

        except Exception as e:
            self.get_logger().error(f"분석 파이프라인 오류: {e}")
        finally:
            self.is_analyzing = False

    # ──────────────────────────────────────────────
    # 헬퍼: 서버에서 주방 좌표 가져오기
    # ──────────────────────────────────────────────

    def _get_kitchen_position(self):
        try:
            url      = f"{self.SERVER_URL}/robots/{self.ROBOT_ID}/zones"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
            zones = response.json().get("zones", [])
            for zone in zones:
                name = zone.get("name", "").lower()
                if self.kitchen_zone_name.lower() in name:
                    return zone.get("center")
        except Exception as e:
            self.get_logger().error(f"주방 좌표 조회 실패: {e}")
        return None

    # ──────────────────────────────────────────────
    # 헬퍼: Nav2로 목표 지점 이동
    # ──────────────────────────────────────────────

    def _navigate_to(self, x: float, y: float, timeout: float = 60.0) -> bool:
        arrived = threading.Event()
        success = [False]

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id    = 'map'
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.position.z    = 0.0
        pose.pose.orientation.w = 1.0
        goal.pose = pose

        def result_callback(future):
            if future.result().status == 4:  # SUCCEEDED
                success[0] = True
            arrived.set()

        def response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                arrived.set()
                return
            goal_handle.get_result_async().add_done_callback(result_callback)

        self.nav_client.send_goal_async(goal).add_done_callback(response_callback)
        arrived.wait(timeout=timeout)
        return success[0]

    # ──────────────────────────────────────────────
    # 헬퍼: /cmd_vel로 제자리 360도 회전
    # ──────────────────────────────────────────────

    def _rotate_360(self, duration_sec: float = 29.0, angular_speed: float = 0.5):
        twist = Twist()
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)
        time.sleep(duration_sec)
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    # ──────────────────────────────────────────────
    # 헬퍼: 클라우드 전송
    # ──────────────────────────────────────────────

    def _upload_to_cloud(self, package: dict):
        headers = {
            "Content-Type": "application/json",
            "X-ARIA-SECRET": self.SECRET_TOKEN
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    self.CLOUD_URL, json=package, headers=headers, timeout=15
                )
                if response.status_code == 200:
                    self.get_logger().info("[Cloud] 전송 성공!")
                    return
                self.get_logger().warn(
                    f"[Cloud] 전송 실패 ({attempt+1}/3): {response.status_code}"
                )
            except Exception as e:
                self.get_logger().warn(f"[Cloud] 네트워크 오류: {e}")
            if attempt < 2:
                time.sleep(2)

        backup_filename = f"fail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(package, f, ensure_ascii=False, indent=4)
        self.get_logger().warn(f"[Cloud] 전송 실패 → 로컬 백업: {backup_path}")


def main(args=None):
    rclpy.init(args=args)
    node = AriaCookingAiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recorder is not None:
            node.recorder.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
