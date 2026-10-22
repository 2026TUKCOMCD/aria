import os
import csv
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Empty
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose


class DataCollectorNode(Node):
    """AI 데이터 수집 + 구역 순회/집중정화(air_purify_scheduler_node 흡수) 노드.

    aria_controller_node의 AI모드 핸드셰이크(/aria/start_purify_cycle →
    /aria/purify_cycle_done)를 그대로 재사용한다 — 독자적인 타이머로
    순회를 시작하지 않고, 컨트롤러가 AI모드일 때만 보내는 트리거에만
    반응한다(기존 air_purify_scheduler_node와 동일한 계약).
    """

    def __init__(self):
        super().__init__('data_collector_node')

        dotenv_path = Path("/srv/aria/users/hs/aria/.env")
        load_dotenv(dotenv_path=dotenv_path, override=True)

        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter('server_url', os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com'))
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))

        # 4단계 분류용 임계값 (air_purify_scheduler_node와 동일 공식/기본값)
        self.declare_parameter('threshold_good', 30.0)
        self.declare_parameter('threshold_bad', 70.0)
        self.declare_parameter('pm25_max', 100.0)
        self.declare_parameter('voc_max', 150.0)

        # 집중정화: 도착 후 팬을 켜고, 회복되거나 최대 시간이 지날 때까지 머문다.
        self.declare_parameter('purify_fan_speed', 255)
        self.declare_parameter('purify_min_sec', 60.0)    # 최소 정화 시간(센서 반응 지연 고려)
        self.declare_parameter('purify_max_sec', 600.0)   # 구역당 최대 정화 시간(10분)

        self._robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self._server_url = str(self.get_parameter('server_url').get_parameter_value().string_value).rstrip('/')
        self._auth_token = self.get_parameter('auth_token').get_parameter_value().string_value
        self._threshold_good = self.get_parameter('threshold_good').get_parameter_value().double_value
        self._threshold_bad = self.get_parameter('threshold_bad').get_parameter_value().double_value
        self._pm25_max = self.get_parameter('pm25_max').get_parameter_value().double_value
        self._voc_max = self.get_parameter('voc_max').get_parameter_value().double_value
        self._purify_fan_speed = int(self.get_parameter('purify_fan_speed').value)
        self._purify_min_sec = float(self.get_parameter('purify_min_sec').value)
        self._purify_max_sec = float(self.get_parameter('purify_max_sec').value)

        # 데이터 저장을 위한 CSV 설정 (기존 data_collector 기능 유지)
        self.save_dir = os.path.expanduser('~/aria_ai_system/data')
        self.csv_path = os.path.join(self.save_dir, 'aria_daily_log.csv')
        os.makedirs(self.save_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['hour', 'minute', 'day_of_week', 'zone_name', 'pm25'])

        # 공기질 원본 데이터 (CSV 기록용 pm25 + 등급 산출용 종합 점수)
        self.current_pm25 = 0.0
        self._current_air_score = -1.0
        self.create_subscription(String, '/esp32/air_raw', self.air_quality_callback, 10)

        # Nav2 Action Client
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_goal_handle = None

        # aria_controller_node 트리거/중단 구독 (air_purify_scheduler_node와 동일 계약)
        self.create_subscription(Empty, '/aria/start_purify_cycle', self.on_start_cycle_trigger, 10)
        self.create_subscription(Empty, '/aria/purify_abort', self.on_purify_abort, 10)

        # 컨트롤러에게 순회/정화 완료를 알리는 퍼블리셔
        self.purify_done_pub = self.create_publisher(Empty, '/aria/purify_cycle_done', 10)

        # 웹앱용 구역별 공기질 등급 업로드 (air_purify_scheduler_node 흡수)
        self.air_quality_pub = self.create_publisher(String, '/aria/zone_air_quality', 10)

        # 집중정화용 팬 제어
        self.fan_pub = self.create_publisher(String, '/esp32/fan_cmd', 10)
        # 청정 시작/완료 알림 → aria_mqtt_node → 웹 로그
        self.event_pub = self.create_publisher(String, '/aria/robot_event', 10)
        self._dwell_started = None      # 현재 구역에서 정화를 시작한 시각
        self._score_before = None       # 정화 시작 시점 점수(개선량 기록용)
        self.create_timer(1.0, self._purify_dwell_tick)

        # 순회/정화 상태
        self.state = 'IDLE'  # IDLE, PATROLLING, PURIFYING
        self.patrol_queue = []
        self.purify_queue = []
        self.measured_scores = []
        self.current_target_zone = None

        self.get_logger().info('🤖 AI 데이터 수집기(Data Collector)가 시작되었습니다.')
        self.get_logger().info(f'기록 경로: {self.csv_path}')

    # ──────────────────────────────────────────────
    # 공기질 콜백
    # ──────────────────────────────────────────────

    def air_quality_callback(self, msg):
        try:
            data = json.loads(msg.data)
            pm25 = float(data.get('pm25', 0.0))
            voc = float(data.get('voc', 0.0))
            self.current_pm25 = pm25

            score = max(pm25 / self._pm25_max, voc / self._voc_max) * 100.0
            self._current_air_score = min(100.0, max(0.0, score))
        except Exception as e:
            self.get_logger().warn(f'⚠️ 공기질 원본 데이터 파싱 실패: {e}')

    # ──────────────────────────────────────────────
    # 컨트롤러 트리거/중단
    # ──────────────────────────────────────────────

    def on_start_cycle_trigger(self, msg):
        self.start_patrol_cycle()

    def on_purify_abort(self, msg):
        if self.state == 'IDLE':
            return
        self.get_logger().warn('🛑 순회/정화가 외부 요청으로 중단되었습니다. IDLE로 전환합니다.')
        self._cancel_current_goal()
        self._end_dwell(fan_off=True)
        self.state = 'IDLE'
        self.patrol_queue.clear()
        self.purify_queue.clear()
        self.measured_scores.clear()
        self.current_target_zone = None

    # ──────────────────────────────────────────────
    # 순회 (데이터 수집 + 공기질 측정)
    # ──────────────────────────────────────────────

    def start_patrol_cycle(self):
        if self.state != 'IDLE':
            self.get_logger().info('⏳ 아직 이전 순회가 끝나지 않았습니다.')
            return

        self.get_logger().info('🔄 [데이터 수집] 구역 순회를 시작합니다.')
        try:
            url = f'{self._server_url}/robots/{self._robot_id}/zones'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token

            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                self.get_logger().error(f'❌ API 서버 응답 오류 (코드: {response.status_code})')
                self.purify_done_pub.publish(Empty())
                return

            zones = response.json().get('zones', [])
            if not zones:
                self.get_logger().warn('⚠️ 등록된 구역 정보가 없습니다.')
                self.purify_done_pub.publish(Empty())
                return

            self.state = 'PATROLLING'
            self.measured_scores.clear()
            self.patrol_queue = zones.copy()
            self.visit_next_zone()

        except Exception as e:
            self.get_logger().error(f'❌ API 구역 데이터 동기화 실패: {e}')
            self.purify_done_pub.publish(Empty())

    def visit_next_zone(self):
        self._current_air_score = -1.0

        if self.state == 'PATROLLING':
            if not self.patrol_queue:
                self.get_logger().info('✅ 모든 구역 순회 및 측정이 완료되었습니다. 데이터를 분류합니다.')
                self.evaluate_and_start_purify()
                return
            target_zone = self.patrol_queue.pop(0)

        elif self.state == 'PURIFYING':
            if not self.purify_queue:
                self.get_logger().info('🎉 모든 나쁨 구역의 집중 정화가 끝났습니다. 대기 상태로 전환합니다.')
                self.state = 'IDLE'
                self.purify_done_pub.publish(Empty())
                return
            target_zone = self.purify_queue.pop(0).get('zone')

        else:
            return

        self.current_target_zone = target_zone
        self.send_navigation_goal(target_zone)

    def send_navigation_goal(self, zone):
        center = zone.get('center', {})
        x, y = center.get('x'), center.get('y')

        if x is None or y is None:
            self.get_logger().warn(f"⚠️ '{zone.get('name')}' 구역의 좌표 정보가 유효하지 않아 제외합니다.")
            self.visit_next_zone()
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        mode_str = "순회/측정" if self.state == 'PATROLLING' else "집중정화"
        self.get_logger().info(f"🚀 [{mode_str}] 목적지 '{zone.get('name')}'(으)로 이동합니다.")

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ 이동 목표가 거부되었습니다. 다음 구역으로 넘어갑니다.')
            self._current_goal_handle = None
            self.visit_next_zone()
            return

        self._current_goal_handle = goal_handle
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self._current_goal_handle = None
        if self.current_target_zone is None or self.state == 'IDLE':
            return
        zone_name = self.current_target_zone.get('name')

        # 도착하지 못했으면 그 구역 값으로 기록·정화하지 않는다.
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f"⚠️ '{zone_name}' 이동 실패(status={status}). 이 구역은 건너뜁니다.")
            self.visit_next_zone()
            return

        if self.state == 'PATROLLING':
            measured_score = self._current_air_score
            self.measured_scores.append({'zone': self.current_target_zone, 'score': measured_score})
            self._write_csv_row(zone_name)
            # 순회로 측정한 모든 구역의 등급을 웹에 올린다.
            # (이전엔 집중정화한 '나쁨' 구역만 올려서 좋음/보통 구역은 웹에서 갱신되지 않았다.
            #  나쁨 구역은 정화가 끝난 뒤 정화 후 점수로 한 번 더 갱신된다.)
            self.upload_air_quality_to_cloud(self.current_target_zone)
            self.visit_next_zone()

        elif self.state == 'PURIFYING':
            # 이전 버전은 도착 즉시 점수를 올리고 다음 구역으로 넘어갔다(정화 없음).
            # 이제는 팬을 켜고 머물며, _purify_dwell_tick이 종료 시점을 판단한다.
            self._score_before = self._current_air_score
            self._dwell_started = time.time()
            self._send_fan(self._purify_fan_speed)
            self._notify_purify(True, zone_name)
            self.get_logger().info(
                f"🌀 '{zone_name}' 집중 정화 시작 (시작 점수 {self._score_before:.1f}, "
                f"최소 {self._purify_min_sec:.0f}s / 최대 {self._purify_max_sec:.0f}s)"
            )

    # ──────────────────────────────────────────────
    # 집중정화 체류
    # ──────────────────────────────────────────────

    def _notify_purify(self, started: bool, zone_name):
        """웹 화면 로그: 청정 시작/완료만 짧게 알린다."""
        msg = String()
        msg.data = json.dumps({
            'type': 'PURIFY_START' if started else 'PURIFY_DONE',
            'message': f'{zone_name} 청정을 시작합니다.' if started else f'{zone_name} 청정을 완료했습니다.',
        }, ensure_ascii=False)
        self.event_pub.publish(msg)

    def _send_fan(self, speed: int):
        msg = String()
        msg.data = json.dumps({'fan': int(speed)})
        self.fan_pub.publish(msg)

    def _end_dwell(self, fan_off: bool):
        if self._dwell_started is not None and fan_off:
            self._send_fan(0)
        self._dwell_started = None

    def _purify_dwell_tick(self):
        if self.state != 'PURIFYING' or self._dwell_started is None:
            return
        elapsed = time.time() - self._dwell_started
        score = self._current_air_score
        recovered = 0 <= score < self._threshold_good

        if elapsed < self._purify_min_sec:
            return
        if not recovered and elapsed < self._purify_max_sec:
            return

        zone_name = self.current_target_zone.get('name')
        reason = '회복' if recovered else '최대 시간 도달'
        self.get_logger().info(
            f"✨ '{zone_name}' 집중 정화 종료({reason}, {elapsed:.0f}s): "
            f"{self._score_before:.1f} → {score:.1f}. 정화 후 결과를 업로드합니다."
        )
        self._end_dwell(fan_off=True)
        self._notify_purify(False, zone_name)
        self.upload_air_quality_to_cloud(self.current_target_zone)
        self.visit_next_zone()

    def _write_csv_row(self, zone_name):
        now = datetime.now()
        try:
            # 웹앱 'AI 초기화'로 CSV가 치워졌을 수 있으므로, 파일이 없으면 헤더부터 다시 쓴다.
            # (헤더 없이 이어 쓰면 night_training의 read_csv가 컬럼을 못 찾는다)
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['hour', 'minute', 'day_of_week', 'zone_name', 'pm25'])
                writer.writerow([now.hour, now.minute, now.weekday(), zone_name, self.current_pm25])
            self.get_logger().info(f"📝 데이터 기록 완료! [{zone_name}] PM2.5: {self.current_pm25:.1f}")
        except Exception as e:
            self.get_logger().error(f"데이터 기록 실패: {e}")

    def evaluate_and_start_purify(self):
        self.get_logger().info('======== 전체 구역 공기질 4단계 평가 결과 ========')

        targets = []
        for item in self.measured_scores:
            z_name = item['zone'].get('name')
            score = item['score']

            if score < 0:
                self.get_logger().warn(f' ❓ [지연] {z_name} : 센서 데이터 미수신')
            elif score >= self._threshold_bad:
                self.get_logger().info(f' 🚨 [나쁨] {z_name} : {score:.1f}점 -> 정화 대상 선정!')
                targets.append(item)
            elif score >= self._threshold_good:
                self.get_logger().info(f' ☁️ [보통] {z_name} : {score:.1f}점 -> 양호')
            else:
                self.get_logger().info(f' 🌿 [좋음] {z_name} : {score:.1f}점 -> 매우 쾌적')

        self.get_logger().info('==================================================')

        if not targets:
            self.get_logger().info('✅ 집중 정화가 요구되는 [나쁨] 단계의 구역이 없습니다. 대기합니다.')
            self.state = 'IDLE'
            self.purify_done_pub.publish(Empty())
            return

        targets.sort(key=lambda x: x['score'], reverse=True)
        self.purify_queue = targets
        self.state = 'PURIFYING'
        self.visit_next_zone()

    def upload_air_quality_to_cloud(self, zone):
        try:
            score_int = int(self._current_air_score)

            if score_int < 0:
                grade = "지연"
            elif score_int >= self._threshold_bad:
                grade = "나쁨"
            elif score_int >= self._threshold_good:
                grade = "보통"
            else:
                grade = "좋음"

            payload = {
                "robot_id": self._robot_id,
                "current_zone": zone.get('name'),
                "air_score": score_int,
                "air_grade": grade,
            }

            msg = String()
            msg.data = json.dumps(payload)
            self.air_quality_pub.publish(msg)

            self.get_logger().info(f"📤 [{grade}] '{zone.get('name')}' 최종 결과({score_int}점) MQTT 노드로 송신 완료")

        except Exception as e:
            self.get_logger().error(f'❌ 공기질 업로드 중 에러 발생: {e}')

    def _cancel_current_goal(self):
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
