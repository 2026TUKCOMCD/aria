import os
import json
import math
import time
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String, Bool, Empty


def build_features(df: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """night_training.build_features와 동일한 규칙(반드시 함께 수정할 것).

    시각·요일은 sin/cos 주기 인코딩, 구역은 one-hot.
    학습 때 저장한 feature_columns 순서로 맞춰서 반환한다.
    """
    minutes = df['hour'].astype(float) * 60.0 + df['minute'].astype(float)
    dow = df['day_of_week'].astype(float)
    feats = pd.DataFrame({
        'tod_sin': np.sin(2 * math.pi * minutes / 1440.0),
        'tod_cos': np.cos(2 * math.pi * minutes / 1440.0),
        'dow_sin': np.sin(2 * math.pi * dow / 7.0),
        'dow_cos': np.cos(2 * math.pi * dow / 7.0),
    }, index=df.index)
    zone_series = df['zone_name'].astype(str)
    for col in feature_columns:
        if col.startswith('zone='):
            feats[col] = (zone_series == col[len('zone='):]).astype(int)
    return feats.reindex(columns=feature_columns, fill_value=0)


class AiCleanerNode(Node):
    """시간대별 구역 오염 확률을 예측해 오염 전에 미리 이동·정화하는 노드.

    변경점
      - 모델: 구역별 이진분류 번들(night_training v2)을 사용. 구역마다 P(오염)을
        따로 계산하고, 임계값을 넘는 구역이 없으면 출동하지 않는다.
      - 핫리로드: 모델 파일이 바뀌면(심야 재학습) 노드 재시작 없이 다시 읽는다.
        구역 좌표는 바뀔 수 있는 시점에만 다시 받는다: 노드 시작, 매핑 종료
        (사용자가 요청한 재매핑으로 구역이 새로 나뉨), 순회 종료.
      - 이동 실패(aborted/canceled)를 도착으로 취급하지 않는다.
      - 정화가 끝나지 않는 경우를 막기 위해 최대 정화 시간을 둔다.
    """

    def __init__(self):
        super().__init__('ai_cleaner_node')

        load_dotenv(dotenv_path=Path("/srv/aria/users/hs/aria/.env"), override=True)

        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter('server_url', os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com'))
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))
        self.declare_parameter('predict_threshold', 0.70)     # 구역별 P(오염) 임계값
        self.declare_parameter('good_pm25', 15.0)             # 이 미만이면 정화 완료
        self.declare_parameter('cooldown_sec', 1800.0)        # 같은 구역 재출동 금지 시간(30분)
        self.declare_parameter('max_purify_sec', 900.0)       # 한 구역 최대 정화 시간(15분)
        # 복귀 위치: 웹에서 저장한 충전(도킹) 위치를 API로 받아 쓴다.
        # home_x/home_y는 도킹 위치를 아직 못 받았을 때만 쓰는 예비값이다.
        self.declare_parameter('home_x', 0.14)
        self.declare_parameter('home_y', 0.00)
        self.declare_parameter(
            'base_api_url',
            os.environ.get('ARIA_BASE_API_URL', os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com')),
        )

        p = lambda n: self.get_parameter(n).value
        self._robot_id = str(p('robot_id'))
        self._server_url = str(p('server_url')).rstrip('/')
        self._base_api_url = str(p('base_api_url')).rstrip('/')
        self._auth_token = p('auth_token')
        self.threshold = float(p('predict_threshold'))
        self.good_threshold = float(p('good_pm25'))
        self.cooldown_duration = float(p('cooldown_sec'))
        self.max_purify_sec = float(p('max_purify_sec'))
        self.home_coords = {'x': float(p('home_x')), 'y': float(p('home_y'))}

        # 구역 좌표 (백그라운드 스레드가 갱신)
        self.zone_dict = {}
        self._zone_lock = threading.Lock()
        self._zone_fetch_running = False
        self._last_mode = None

        # 모델 번들 (파일 mtime이 바뀌면 다시 로드)
        self.model_path = os.path.join(os.path.expanduser('~/aria_ai_system/models'), 'aria_rf_model.pkl')
        self.bundle = None
        self._model_mtime = None
        self._warned_unknown_zones = set()

        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_goal_handle = None
        self.fan_pub = self.create_publisher(String, '/esp32/fan_cmd', 10)
        # 청정 시작/완료 알림 → aria_mqtt_node → 웹 로그
        self.event_pub = self.create_publisher(String, '/aria/robot_event', 10)
        self.create_subscription(String, '/esp32/air_raw', self.air_callback, 10)

        self._ai_active = False
        self.create_subscription(Bool, '/aria/ai_active', self.ai_active_callback, 10)

        # 구역이 바뀔 수 있는 시점에만 다시 조회한다(주기 폴링 없음).
        self.create_subscription(String, '/aria/mode_state', self._on_mode_state, 10)
        self.create_subscription(Empty, '/aria/purify_cycle_done', lambda _m: self._refresh_zones_async(), 10)

        self.is_moving = False
        self.is_purifying = False
        self.purify_started_at = None
        self.current_target_zone = 'home'
        self.cooldown_dict = {}

        self._refresh_zones_async()
        self._maybe_reload_model()
        self.create_timer(10.0, self.predict_and_move)

        self.get_logger().info('🤖 AI 예측 청정기(AI Cleaner)가 대기 중입니다.')

    # ──────────────────────────────────────────────
    # 구역 / 모델 갱신
    # ──────────────────────────────────────────────

    def _on_mode_state(self, msg: String):
        mode = msg.data
        if self._last_mode == 'MAPPING' and mode != 'MAPPING':
            self.get_logger().info('매핑 종료 감지 — 새 구역 목록을 받습니다.')
            self._refresh_zones_async()
        self._last_mode = mode

    def _refresh_zones_async(self):
        if self._zone_fetch_running:
            return
        self._zone_fetch_running = True
        threading.Thread(target=self._fetch_zone_coordinates, daemon=True).start()

    def _fetch_zone_coordinates(self):
        try:
            url = f'{self._server_url}/robots/{self._robot_id}/zones'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                self.get_logger().warn(f'구역 조회 실패 (HTTP {response.status_code}) — 기존 좌표 유지')
                return
            new_dict = {}
            for z in response.json().get('zones', []):
                name, center = z.get('name'), z.get('center')
                if name and center and center.get('x') is not None and center.get('y') is not None:
                    new_dict[str(name)] = center
            with self._zone_lock:
                changed = set(new_dict) != set(self.zone_dict)
                self.zone_dict = new_dict
            if changed:
                self.get_logger().info(f'🗺️ 구역 목록 갱신: {sorted(new_dict)}')
        except Exception as e:
            self.get_logger().warn(f'구역 조회 오류: {e} — 기존 좌표 유지')
        finally:
            # 도킹 위치도 구역과 같은 시점(시작·매핑 종료·순회 종료)에 바뀔 수 있다.
            self._fetch_home_coordinates()
            self._zone_fetch_running = False

    def _fetch_home_coordinates(self):
        """저장된 충전(도킹) 위치를 복귀 좌표로 쓴다. 실패하면 기존 값을 유지한다."""
        try:
            url = f'{self._base_api_url}/robots/{self._robot_id}/dock'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                return
            result = response.json()
            data = result.get('data') if isinstance(result, dict) else None
            if result.get('success') is not True or not isinstance(data, dict):
                return
            home = {'x': float(data['x']), 'y': float(data['y'])}
            if home != self.home_coords:
                self.home_coords = home
                self.get_logger().info(f"🏠 복귀 위치(도킹) 갱신: x={home['x']:.2f}, y={home['y']:.2f}")
        except Exception as e:
            self.get_logger().warn(f'도킹 위치 조회 실패: {e} — 기존 복귀 좌표 유지')

    def _maybe_reload_model(self):
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            # 모델 파일이 없다 — 아직 학습 전이거나, 웹앱 'AI 초기화'로 치워진 경우.
            # 메모리에 남은 옛 모델로 계속 예측하지 않도록 내려놓는다.
            if self.bundle is not None:
                self.get_logger().warn('모델 파일이 사라져 예측을 멈춥니다(AI 초기화). 새 학습을 기다립니다.')
            self.bundle = None
            self._model_mtime = None
            return
        if mtime == self._model_mtime:
            return
        self._model_mtime = mtime
        try:
            bundle = joblib.load(self.model_path)
        except Exception as e:
            self.get_logger().error(f'모델 로드 실패: {e}')
            return
        if not isinstance(bundle, dict) or bundle.get('version', 1) < 2:
            # 구버전(오염 행만으로 구역을 맞히던 모델)은 의미가 다르므로 쓰지 않는다.
            self.get_logger().warn('구버전 모델 형식입니다. 새 학습 결과가 나올 때까지 예측을 쉽니다.')
            self.bundle = None
            return
        self.bundle = bundle
        self._warned_unknown_zones.clear()
        self.get_logger().info(
            f"🧠 모델 로드: 학습 {bundle.get('trained_at')} | 오염 {bundle.get('n_positive')}건 "
            f"| 평가 {bundle.get('metrics')}"
        )

    # ──────────────────────────────────────────────
    # 활성/팬/복귀
    # ──────────────────────────────────────────────

    def ai_active_callback(self, msg: Bool):
        self._ai_active = bool(msg.data)
        if not self._ai_active:
            self.get_logger().info("⏸️ 예측 청정 대기 상태로 전환합니다.")
            self._cancel_current_goal()
            self._stop_purifying(start_cooldown=False)
            self.is_moving = False

    def _cancel_current_goal(self):
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None

    def _notify_purify(self, started: bool, zone_name):
        """웹 화면 로그: 청정 시작/완료만 짧게 알린다."""
        msg = String()
        msg.data = json.dumps({
            'type': 'PURIFY_START' if started else 'PURIFY_DONE',
            'message': f'{zone_name} 청정을 시작합니다.' if started else f'{zone_name} 청정을 완료했습니다.',
        }, ensure_ascii=False)
        self.event_pub.publish(msg)

    def send_fan_cmd(self, fan_speed: int):
        msg = String()
        msg.data = json.dumps({"fan": fan_speed})
        self.fan_pub.publish(msg)
        self.get_logger().info(f"💨 팬 가동 상태 변경: {fan_speed}")

    def _stop_purifying(self, start_cooldown: bool):
        if start_cooldown and self.current_target_zone not in (None, 'home'):
            self.cooldown_dict[self.current_target_zone] = time.time()
        if self.is_purifying:
            self.send_fan_cmd(0)
            if start_cooldown:  # 정상 종료(회복·최대시간·구역 전환)만 알린다. AI 끄기/취침으로 멈춘 경우는 제외
                self._notify_purify(False, self.current_target_zone)
        self.is_purifying = False
        self.purify_started_at = None

    def return_to_home(self):
        if self.is_moving or self.current_target_zone == 'home':
            return
        self.get_logger().info("🏠 충전 대기소(Home)로 복귀합니다.")
        self.current_target_zone = 'home'
        self.send_navigation_goal('home', custom_coords=self.home_coords)

    def _in_cooldown(self, zone_name):
        return time.time() - self.cooldown_dict.get(zone_name, 0.0) < self.cooldown_duration

    # ──────────────────────────────────────────────
    # 공기질 콜백 (정화 종료 판단)
    # ──────────────────────────────────────────────

    def air_callback(self, msg: String):
        if not self._ai_active or not self.is_purifying:
            return
        try:
            pm25 = float(json.loads(msg.data).get('pm25', -1.0))
        except Exception:
            return
        if 0 <= pm25 < self.good_threshold:
            self.get_logger().info(f"✨ 공기질 회복 (PM2.5: {pm25}). 사전 청정을 종료합니다.")
            self._stop_purifying(start_cooldown=True)
            self.return_to_home()

    # ──────────────────────────────────────────────
    # 예측
    # ──────────────────────────────────────────────

    def predict_zone_probs(self, now: datetime) -> dict:
        """현재 등록된 구역마다 P(오염)을 계산한다."""
        with self._zone_lock:
            zone_names = list(self.zone_dict)
        model_zones = set(self.bundle['zones'])

        unknown = [z for z in zone_names if z not in model_zones]
        new_unknown = set(unknown) - self._warned_unknown_zones
        if new_unknown:
            # 재매핑·이름 변경으로 학습 데이터에 없는 구역. 추측하지 않고 건너뛴다.
            self.get_logger().warn(f'학습 데이터에 없는 구역은 예측에서 제외: {sorted(new_unknown)}')
            self._warned_unknown_zones |= new_unknown

        known = [z for z in zone_names if z in model_zones]
        if not known:
            return {}
        rows = pd.DataFrame([{
            'hour': now.hour, 'minute': now.minute,
            'day_of_week': now.weekday(), 'zone_name': z,
        } for z in known])
        X = build_features(rows, self.bundle['feature_columns'])
        model = self.bundle['model']
        pos_idx = list(model.classes_).index(1)
        probs = model.predict_proba(X)[:, pos_idx]
        return dict(zip(known, probs))

    def predict_and_move(self):
        self._maybe_reload_model()

        if not self._ai_active:
            return

        # 정화가 너무 오래 걸리면 끊는다(센서 고장·외부 오염원 지속 대비).
        if self.is_purifying and self.purify_started_at is not None:
            if time.time() - self.purify_started_at > self.max_purify_sec:
                self.get_logger().warn(f"⏱️ 최대 정화 시간({self.max_purify_sec:.0f}s) 초과. 종료합니다.")
                self._stop_purifying(start_cooldown=True)
                self.return_to_home()
                return

        if self.is_moving or self.bundle is None or not self.zone_dict:
            return

        now = datetime.now()
        try:
            probs = self.predict_zone_probs(now)
        except Exception as e:
            self.get_logger().error(f'예측 실패: {e}')
            return

        candidates = {z: p for z, p in probs.items()
                      if p >= self.threshold and not self._in_cooldown(z)}

        if not candidates:
            # 오염이 예상되는 구역이 없으면 출동하지 않는다(정화 중이면 마저 한다).
            if not self.is_purifying and self.current_target_zone != 'home':
                self.return_to_home()
            return

        target, prob = max(candidates.items(), key=lambda kv: kv[1])
        if target == self.current_target_zone:
            return  # 이미 그 구역에서 정화 중

        if self.is_purifying:
            # 다른 구역의 오염 확률이 더 높아졌다. 현재 구역 정화를 끝내고 이동한다.
            self._stop_purifying(start_cooldown=True)

        self.current_target_zone = target
        self.get_logger().info(
            f"🚨 예측 [{now:%H:%M}] '{target}' 오염 확률 {prob * 100:.1f}% → 선제 이동 "
            f"(전체: { {z: round(float(v), 2) for z, v in probs.items()} })"
        )
        self.send_navigation_goal(target)

    # ──────────────────────────────────────────────
    # 이동
    # ──────────────────────────────────────────────

    def send_navigation_goal(self, zone_name, custom_coords=None):
        if custom_coords:
            target_coords = custom_coords
        else:
            with self._zone_lock:
                target_coords = self.zone_dict.get(zone_name)
        if not target_coords:
            return

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 액션 서버 응답 없음 — 다음 주기에 다시 시도합니다.')
            self.current_target_zone = None
            return

        self.is_moving = True
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target_coords.get('x', 0.0))
        goal_msg.pose.pose.position.y = float(target_coords.get('y', 0.0))
        goal_msg.pose.pose.orientation.w = 1.0

        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('이동 목표가 거부되었습니다.')
            self.is_moving = False
            self._current_goal_handle = None
            self.current_target_zone = None
            return
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.is_moving = False
        self._current_goal_handle = None
        status = future.result().status

        if status != GoalStatus.STATUS_SUCCEEDED:
            # 도착하지 못했는데 팬을 켜면 엉뚱한 곳을 정화하게 된다.
            failed = self.current_target_zone
            self.get_logger().warn(f"⚠️ '{failed}' 이동 실패(status={status}). 정화하지 않습니다.")
            if failed not in (None, 'home'):
                self.cooldown_dict[failed] = time.time()  # 같은 곳으로 계속 재시도하지 않게
            self.current_target_zone = None
            return

        if self.current_target_zone == 'home':
            self.get_logger().info('🔌 충전 대기소 도착. 다음 예측을 대기합니다.')
        elif self._ai_active:
            self.get_logger().info(f"🏁 '{self.current_target_zone}' 도착. 사전 청정을 시작합니다.")
            self.is_purifying = True
            self.purify_started_at = time.time()
            self.send_fan_cmd(255)
            self._notify_purify(True, self.current_target_zone)


def main(args=None):
    rclpy.init(args=args)
    node = AiCleanerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 종료 신호 수신. 팬을 정지합니다...")
        node.send_fan_cmd(0)
        time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
