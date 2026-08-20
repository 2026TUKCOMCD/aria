#!/usr/bin/env python3
import json
import math
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, time as dt_time
from pathlib import Path

import requests
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose
from explore_lite_msgs.msg import ExploreStatus
from dotenv import load_dotenv

# .env 파일 로드 (다른 ARIA 노드들과 동일한 경로 관례)
_dotenv_path = Path("/srv/aria/users/hs/aria/.env")
load_dotenv(dotenv_path=_dotenv_path, override=True)


class AriaControllerNode(Node):
    """ARIA 메인 통합(상태머신) 노드.

    "모든 노드를 컨트롤하는 메인 통합 노드" 그 자체 — 요리감지 AI는
    aria_cooking_ai_node로 분리되어 여기엔 torch/YOLO 의존성이 없다.

    모드: STANDBY(대기) / AI(순회+요리감지) / MAPPING(매핑) / BASIC(기본모드)
    초기 상태는 항상 STANDBY — 부팅 즉시 순회를 시작하지 않는다.

    이 노드가 유일하게 소유하는 것:
      - 수동이동/기본모드/매핑귀환용 NavigateToPose ActionClient
        (patrol 전용 ActionClient는 air_purify_scheduler_node,
         주방이동 전용 ActionClient는 aria_cooking_ai_node가 각자 소유 —
         활성 구간이 상태머신에 의해 서로 겹치지 않도록 보장된다)
      - AMCL(LOCALIZE) <-> SLAM+explore_lite(MAP) 서브프로세스 스왑
        (ROS2 launch는 실행 중인 트리 일부만 재시작할 수 없어서 직접
         `ros2 launch`를 subprocess로 관리한다 — "launch 상시 실행" 원칙의
         유일한 예외)
    """

    def __init__(self):
        super().__init__('aria_controller_node')

        # ── 파라미터 ─────────────────────────────────────
        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter('server_url', os.environ.get('ARIA_API_URL', 'http://localhost:5000'))
        self.declare_parameter(
            'base_api_url',
            os.environ.get('ARIA_BASE_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com'),
        )
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))

        self.declare_parameter('purify_interval', 14400.0)  # AI모드 순회 재실행 주기 (4시간)

        # 기본모드 점수 계산 (air_purify_scheduler_node와 동일 공식/기본값)
        self.declare_parameter('threshold_good', 30.0)
        self.declare_parameter('pm25_max', 100.0)
        self.declare_parameter('voc_max', 150.0)

        self.declare_parameter('map_save_dir', os.environ.get('ARIA_MAPS_DIR', '/srv/aria/users/hs/ros2_last/maps'))
        self.declare_parameter(
            'upload_map_script_path',
            os.environ.get('ARIA_UPLOAD_MAP_SCRIPT', '/srv/aria/users/hs/ros2_last/src/upload_map_with_zones.py'),
        )
        self.declare_parameter('mapping_timeout_sec', 3600.0)
        self.declare_parameter('nav_timeout_sec', 120.0)

        self.robot_id = self.get_parameter('robot_id').value
        self.server_url = str(self.get_parameter('server_url').value).rstrip('/')
        self.base_api_url = str(self.get_parameter('base_api_url').value).rstrip('/')
        self.auth_token = self.get_parameter('auth_token').value
        self.purify_interval = float(self.get_parameter('purify_interval').value)
        self.threshold_good = float(self.get_parameter('threshold_good').value)
        self._pm25_max = float(self.get_parameter('pm25_max').value)
        self._voc_max = float(self.get_parameter('voc_max').value)
        self.map_save_dir = self.get_parameter('map_save_dir').value
        self.upload_map_script_path = self.get_parameter('upload_map_script_path').value
        self.mapping_timeout_sec = float(self.get_parameter('mapping_timeout_sec').value)
        self.nav_timeout_sec = float(self.get_parameter('nav_timeout_sec').value)

        # ── 상태 ─────────────────────────────────────────
        self.mode = 'STANDBY'
        self._current_goal_handle = None
        self._nav_goal_lock = threading.Lock()
        self._purify_done_event = threading.Event()
        self._explore_finished_event = threading.Event()
        self._ai_purify_timer_start = None
        self._purify_retrigger_in_progress = False
        self._current_air_score = -1.0
        self._tier2_proc = None
        self._tier2_profile = None

        # ── Nav2 액션 클라이언트 (수동이동/기본모드/매핑귀환 전용) ──
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── 서비스 클라이언트 ────────────────────────────
        self.save_dock_client = self.create_client(Trigger, '/save_dock_pose')
        self.reinit_client = self.create_client(Trigger, '/reinitialize_amcl')

        # ── 구독 ─────────────────────────────────────────
        self.create_subscription(String, '/aria/mode_command', self._on_mode_command, 10)
        self.create_subscription(String, '/aria/manual_move_command', self._on_manual_move_command, 10)
        self.create_subscription(Empty, '/aria/purify_cycle_done', self._on_purify_cycle_done, 10)

        # explore_lite가 이미 발행하는 탐사 상태 토픽 (explore_lite_msgs/ExploreStatus).
        # EXPLORATION_COMPLETE가 오면 매핑이 끝난 것으로 간주한다.
        explore_status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(ExploreStatus, 'explore/status', self._on_explore_status, explore_status_qos)

        self.create_subscription(String, '/esp32/air_raw', self._on_air_raw, 10)

        # ── 발행 ─────────────────────────────────────────
        self.mode_state_pub = self.create_publisher(String, '/aria/mode_state', 10)
        self.ai_active_pub = self.create_publisher(Bool, '/aria/ai_active', 10)
        self.nav_event_pub = self.create_publisher(String, '/aria/nav_event', 10)
        self.purify_trigger_pub = self.create_publisher(Empty, '/aria/start_purify_cycle', 10)
        self.purify_abort_pub = self.create_publisher(Empty, '/aria/purify_abort', 10)
        self.motor_cmd_pub = self.create_publisher(String, '/esp32/motor_cmd', 10)

        # AI모드 4시간 재실행 감시 (1초 주기)
        self.create_timer(1.0, self._ai_mode_tick)

        # 기본값: 위치추정(AMCL) 스택을 항상 띄워둔다
        self._swap_tier2('LOCALIZE')
        self._publish_mode_state()

        self.get_logger().info(f'ARIA Controller Node 시작 [Robot ID: {self.robot_id}] mode=STANDBY')

    # ══════════════════════════════════════════════════
    # 모드 명령 디스패치
    # ══════════════════════════════════════════════════

    def _on_mode_command(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {'mode': msg.data.strip()}

        mode = str(payload.get('mode', '')).strip().upper()

        if self.mode == 'MAPPING' and mode != 'MAPPING':
            self.get_logger().warn('매핑 진행 중에는 모드 전환 요청을 무시합니다.')
            return

        # AI 순회 중에는 STANDBY(명시적 중단)로만 빠져나올 수 있다.
        # 그 외 명령(MAPPING 등)이 끼어들어 순회가 조용히 끊기는 것을 막는다.
        if self.mode == 'AI' and mode not in ('AI', 'AI_MODE', 'AI모드', 'STANDBY', '대기', '대기모드', 'WAIT'):
            self.get_logger().warn(f'AI 모드 진행 중에는 STANDBY로 먼저 전환해야 합니다. (요청: {mode!r})')
            return

        if mode in ('STANDBY', '대기', '대기모드', 'WAIT'):
            self._enter_standby()
        elif mode in ('AI', 'AI_MODE', 'AI모드'):
            self._enter_ai_mode()
        elif mode in ('MAPPING', '매핑', 'SLAM'):
            self._enter_mapping()
        elif mode in ('BASIC', '기본모드'):
            self._enter_basic_mode(payload)
        else:
            self.get_logger().warn(f'알 수 없는 모드 명령: {mode!r}')

    # ══════════════════════════════════════════════════
    # STANDBY
    # ══════════════════════════════════════════════════

    def _enter_standby(self):
        self.mode = 'STANDBY'
        self._publish_mode_state()
        self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._set_ai_active(False)
        self._publish_motor_stop()

    # ══════════════════════════════════════════════════
    # AI모드
    # ══════════════════════════════════════════════════

    def _enter_ai_mode(self):
        if self.mode == 'AI':
            return
        self._cancel_own_nav_goal()
        self.mode = 'AI'
        self._ai_purify_timer_start = None
        self._publish_mode_state()
        #threading.Thread(target=self._ai_mode_worker, daemon=True).start()

    def _ai_mode_worker(self):
        self._run_purify_cycle_and_wait()
        if self.mode != 'AI':
            return  # 대기중 다른 모드로 전환됨
        self._ai_purify_timer_start = time.time()
        self._set_ai_active(True)

    def _run_purify_cycle_and_wait(self, timeout: float = 600.0):
        self._purify_done_event.clear()
        self.purify_trigger_pub.publish(Empty())
        self._purify_done_event.wait(timeout=timeout)

    def _ai_mode_tick(self):
        if self.mode != 'AI':
            return
        if self._ai_purify_timer_start is None:
            return  # 최초 순회가 아직 안 끝남
        if self._purify_retrigger_in_progress:
            return
        elapsed = time.time() - self._ai_purify_timer_start
        if elapsed < self.purify_interval:
            return
        if self._is_night_time():
            return  # 야간엔 재실행을 미루고 다음 틱에 다시 확인
        self._purify_retrigger_in_progress = True
        self._set_ai_active(False)
        threading.Thread(target=self._rerun_purify_cycle, daemon=True).start()

    def _rerun_purify_cycle(self):
        try:
            self._run_purify_cycle_and_wait()
            if self.mode != 'AI':
                return
            self._ai_purify_timer_start = time.time()
            self._set_ai_active(True)
        finally:
            self._purify_retrigger_in_progress = False

    def _on_purify_cycle_done(self, msg: Empty):
        self._purify_done_event.set()

    @staticmethod
    def _is_night_time() -> bool:
        now = datetime.now().time()
        start, end = dt_time(23, 0), dt_time(7, 0)
        if start <= end:
            return start <= now <= end
        return now >= start or now <= end

    # ══════════════════════════════════════════════════
    # 매핑
    # ══════════════════════════════════════════════════

    def _enter_mapping(self):
        if self.mode == 'MAPPING':
            return
        self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._set_ai_active(False)
        self.mode = 'MAPPING'
        self._publish_mode_state()
        threading.Thread(target=self._mapping_worker, daemon=True).start()

    def _mapping_worker(self):
        try:
            dock_pose = self._get_dock_pose()

            self._swap_tier2('MAP')

            self._explore_finished_event.clear()
            finished = self._explore_finished_event.wait(timeout=self.mapping_timeout_sec)
            if not finished:
                self.get_logger().error('매핑이 제한 시간 내에 끝나지 않았습니다.')
                return

            if dock_pose is not None:
                arrived = self._navigate_to(
                    dock_pose['x'], dock_pose['y'], dock_pose.get('theta', 0.0),
                    timeout=self.nav_timeout_sec,
                    metadata={'target_type': 'MAPPING_RETURN'},
                )
                if arrived:
                    self._call_trigger_service_sync(self.save_dock_client)
            else:
                self.get_logger().warn('저장된 도킹 좌표가 없어 매핑 귀환을 건너뜁니다.')

            map_path = self._save_map()
            if map_path:
                self._upload_map_with_zones(map_path)

            self._swap_tier2('LOCALIZE', map_yaml=f'{map_path}.yaml' if map_path else None)
            self._call_trigger_service_sync(self.reinit_client)

        except Exception as e:
            self.get_logger().error(f'매핑 시퀀스 오류: {e}')
        finally:
            if self.mode == 'MAPPING':
                self._enter_standby()

    def _on_explore_status(self, msg: ExploreStatus):
        if msg.status == ExploreStatus.EXPLORATION_COMPLETE:
            self._explore_finished_event.set()

    def _save_map(self):
        map_id = 'aria_map'
        version = int(time.time())
        save_path = os.path.join(self.map_save_dir, f'{map_id}_{version}')
        cmd = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', save_path]
        self.get_logger().info(f'[매핑] 맵 저장: {" ".join(cmd)}')
        try:
            result = subprocess.run(cmd, timeout=30)
            if result.returncode == 0:
                return save_path
            self.get_logger().error(f'맵 저장 명령이 실패했습니다 (returncode={result.returncode})')
        except Exception as e:
            self.get_logger().error(f'맵 저장 실패: {e}')
        return None

    def _upload_map_with_zones(self, map_path: str):
        cmd = ['python3', self.upload_map_script_path, map_path]
        self.get_logger().info(f'[매핑] 구역분할+업로드: {" ".join(cmd)}')
        try:
            subprocess.run(cmd, timeout=120)
        except Exception as e:
            self.get_logger().error(f'맵 업로드 스크립트 실행 실패: {e}')

    # ══════════════════════════════════════════════════
    # 기본모드
    # ══════════════════════════════════════════════════

    def _enter_basic_mode(self, payload: dict):
        self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._set_ai_active(False)
        self.mode = 'BASIC'
        self._publish_mode_state()
        threading.Thread(target=self._basic_mode_worker, args=(payload,), daemon=True).start()

    def _basic_mode_worker(self, payload: dict):
        target = self._resolve_basic_target(payload)
        if target is None or target.get('x') is None or target.get('y') is None:
            # 기본모드는 수동모드 — 좌표 없이 들어오면 에러 없이 그 자리에서
            # 대기하며 수동 이동(/aria/manual_move_command)만 받는다.
            self.get_logger().info('기본모드(수동) 대기 중 — 좌표 지정 후 청정 시작 명령을 기다립니다.')
            return

        arrived = self._navigate_to(
            target['x'], target['y'], target.get('theta', 0.0),
            timeout=self.nav_timeout_sec,
            metadata={'target_type': 'BASIC', 'zone_name': target.get('name', '기본모드 목표')},
        )
        if not arrived or self.mode != 'BASIC':
            if self.mode == 'BASIC':
                self.get_logger().warn('기본모드 목표 이동 실패. 대기모드로 복귀합니다.')
                self._enter_standby()
            return

        self._current_air_score = -1.0
        while self.mode == 'BASIC':
            if 0.0 <= self._current_air_score <= self.threshold_good:
                break
            time.sleep(5.0)

        if self.mode == 'BASIC':
            self.get_logger().info('기본모드 정화 완료(공기질 양호). 대기모드로 복귀합니다.')
            self._enter_standby()

    def _resolve_basic_target(self, payload: dict):
        if 'x' in payload and 'y' in payload:
            try:
                return {
                    'x': float(payload['x']),
                    'y': float(payload['y']),
                    'theta': float(payload.get('theta', 0.0)),
                    'name': payload.get('zone_name', '지정 위치'),
                }
            except (TypeError, ValueError):
                self.get_logger().warn('기본모드 x/y 파싱 실패, 다른 방법으로 목표를 찾습니다.')

        zone_id = payload.get('zone_id')
        if zone_id is not None:
            zone = self._find_zone(zone_id)
            if zone is not None:
                center = zone.get('center', {})
                return {'x': center.get('x'), 'y': center.get('y'), 'theta': 0.0, 'name': zone.get('name')}

        # 좌표/구역이 명시되지 않으면 목표 없음 — 도킹 좌표로 임의 이동하지 않고
        # 기본모드(수동)는 그냥 대기 상태를 유지한다.
        return None

    def _find_zone(self, zone_id):
        try:
            url = f'{self.server_url}/robots/{self.robot_id}/zones'
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
            for zone in response.json().get('zones', []):
                if str(zone.get('id')) == str(zone_id) or zone.get('name') == zone_id:
                    return zone
        except Exception as e:
            self.get_logger().error(f'구역 조회 실패: {e}')
        return None

    def _on_air_raw(self, msg: String):
        if self.mode != 'BASIC':
            return
        try:
            data = json.loads(msg.data)
            pm25 = float(data.get('pm25', 0.0))
            voc = float(data.get('voc', 0.0))
            score = max(pm25 / self._pm25_max, voc / self._voc_max) * 100.0
            self._current_air_score = min(100.0, max(0.0, score))
        except Exception as e:
            self.get_logger().warn(f'기본모드 공기질 파싱 실패: {e}')

    # ══════════════════════════════════════════════════
    # 수동 이동 (MQTT MOVE_TO 릴레이)
    # ══════════════════════════════════════════════════

    def _on_manual_move_command(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'수동 이동 명령 파싱 실패: {e}')
            return

        if self.mode == 'MAPPING':
            self.get_logger().warn('매핑 중에는 수동 이동 명령을 무시합니다.')
            return

        cmd_type = str(payload.get('type', 'MOVE_TO')).upper()
        if cmd_type in ('CANCEL', 'STOP'):
            self._cancel_own_nav_goal()
            return

        x, y = payload.get('x'), payload.get('y')
        if x is None or y is None:
            self.get_logger().error('수동 이동 명령에 x/y가 없습니다.')
            return
        theta = float(payload.get('theta', 0.0))

        # 순회/AI 중이었다면 수동 이동을 우선하고 자동 동작은 중단
        if self.mode == 'AI':
            self.purify_abort_pub.publish(Empty())
            self._set_ai_active(False)
        self._cancel_own_nav_goal()

        metadata = {k: payload[k] for k in ('target_type', 'zone_id', 'zone_name', 'target_name') if k in payload}
        metadata.update({'x': float(x), 'y': float(y), 'theta': theta})

        threading.Thread(
            target=self._manual_move_worker, args=(float(x), float(y), theta, metadata), daemon=True
        ).start()

    def _manual_move_worker(self, x: float, y: float, theta: float, metadata: dict):
        arrived = self._navigate_to(x, y, theta, timeout=self.nav_timeout_sec, metadata=metadata)
        if arrived and str(metadata.get('target_type', '')).upper() == 'DOCK_SETUP':
            self._call_trigger_service_sync(self.save_dock_client)

    # ══════════════════════════════════════════════════
    # Nav2 헬퍼 (수동이동/기본모드/매핑귀환 공용)
    # ══════════════════════════════════════════════════

    def _navigate_to(self, x: float, y: float, theta: float = 0.0, timeout: float = 60.0, metadata: dict = None) -> bool:
        metadata = metadata or {}

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self._publish_nav_event('NAVIGATION_FAILED', metadata, reason='Nav2 서버를 찾을 수 없습니다')
            return False

        arrived = threading.Event()
        success = [False]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        half_yaw = theta / 2.0
        goal.pose.pose.orientation.z = math.sin(half_yaw)
        goal.pose.pose.orientation.w = math.cos(half_yaw)

        def result_cb(future):
            result = future.result()
            if result is not None and result.status == 4:  # SUCCEEDED
                success[0] = True
            arrived.set()

        def response_cb(future):
            goal_handle = future.result()
            accepted = goal_handle is not None and goal_handle.accepted
            with self._nav_goal_lock:
                self._current_goal_handle = goal_handle if accepted else None
            if not accepted:
                arrived.set()
                return
            goal_handle.get_result_async().add_done_callback(result_cb)

        self._publish_nav_event('NAVIGATION_REQUESTED', metadata)
        self.nav_client.send_goal_async(goal).add_done_callback(response_cb)
        arrived.wait(timeout=timeout)

        with self._nav_goal_lock:
            self._current_goal_handle = None

        event = 'NAVIGATION_SUCCEEDED' if success[0] else 'NAVIGATION_FAILED'
        self._publish_nav_event(event, metadata)
        return success[0]

    def _cancel_own_nav_goal(self):
        with self._nav_goal_lock:
            goal_handle = self._current_goal_handle
        if goal_handle is not None:
            goal_handle.cancel_goal_async()

    def _publish_nav_event(self, event: str, metadata: dict, **extra):
        payload = {'event': event, **metadata, **extra}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.nav_event_pub.publish(msg)

    # ══════════════════════════════════════════════════
    # 서비스 헬퍼
    # ══════════════════════════════════════════════════

    def _call_trigger_service_sync(self, client, timeout: float = 15.0):
        # Tier2를 LOCALIZE로 스왑한 직후(_swap_tier2)에는 11개 노드가 한꺼번에
        # 기동하며 라즈베리파이의 CPU/DDS 디스커버리가 잠깐 포화되어, 계속 떠있던
        # Tier1 서비스(charging_pose_initializer의 /reinitialize_amcl 등)조차
        # 몇 초 안에 못 찾는 경우가 있었다(5초 타임아웃에서 실측 재현됨). 20초로 늘림.
        if not client.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(f'{client.srv_name} 서비스를 찾을 수 없습니다.')
            return False

        done = threading.Event()
        result_holder = {}

        def _cb(future):
            try:
                resp = future.result()
                result_holder['success'] = resp.success
                result_holder['message'] = resp.message
            except Exception as e:
                result_holder['success'] = False
                result_holder['message'] = str(e)
            done.set()

        client.call_async(Trigger.Request()).add_done_callback(_cb)
        done.wait(timeout=timeout)

        success = result_holder.get('success', False)
        message = result_holder.get('message', '서비스 응답 시간 초과')
        if success:
            self.get_logger().info(f'{client.srv_name}: {message}')
        else:
            self.get_logger().error(f'{client.srv_name} 실패: {message}')
        return success

    # ══════════════════════════════════════════════════
    # 도킹 좌표 조회 (charging_pose_initializer와 동일 API)
    # ══════════════════════════════════════════════════

    def _get_dock_pose(self):
        try:
            url = f'{self.base_api_url}/robots/{self.robot_id}/dock'
            headers = {'Accept': 'application/json'}
            if self.auth_token:
                headers['Authorization'] = self.auth_token
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                return None
            result = response.json()
            if result.get('success') is not True:
                return None
            data = result.get('data')
            if not isinstance(data, dict):
                return None
            return {'x': float(data['x']), 'y': float(data['y']), 'theta': float(data['theta'])}
        except Exception as e:
            self.get_logger().error(f'도킹 좌표 조회 실패: {e}')
            return None

    # ══════════════════════════════════════════════════
    # Tier2 (AMCL <-> SLAM) 서브프로세스 관리
    # ══════════════════════════════════════════════════

    def _swap_tier2(self, profile: str, map_yaml: str = None):
        self._stop_tier2()

        if profile == 'LOCALIZE':
            # ❌ 여기 패키지 이름이 my_robot_bringup으로 되어 있음
            cmd = ['ros2', 'launch', 'my_robot_bringup', 'localize.launch.py']
            if map_yaml:
                cmd.append(f'map_yaml:={map_yaml}')
        else:  # 'MAP'
            # ❌ 여기도 패키지 이름이 my_robot_bringup으로 되어 있음
            cmd = ['ros2', 'launch', 'my_robot_bringup', 'mapping.launch.py', 'return_to_init:=false']
        self.get_logger().info(f'[Tier2] {profile} 프로필 기동: {" ".join(cmd)}')
        
        try:
            self._tier2_proc = subprocess.Popen(cmd)
            self._tier2_profile = profile
        except Exception as e:
            self.get_logger().error(f'[Tier2] {profile} 기동 실패: {e}')
            self._tier2_proc = None
            self._tier2_profile = None

    def _stop_tier2(self):
        if self._tier2_proc is None:
            return
        if self._tier2_proc.poll() is not None:
            self._tier2_proc = None
            return

        self.get_logger().info(f'[Tier2] {self._tier2_profile} 프로필 종료 중...')
        self._tier2_proc.send_signal(signal.SIGINT)
        try:
            self._tier2_proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self._tier2_proc.terminate()
            try:
                self._tier2_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._tier2_proc.kill()
        self._tier2_proc = None

    # ══════════════════════════════════════════════════
    # 기타 헬퍼
    # ══════════════════════════════════════════════════

    def _set_ai_active(self, active: bool):
        msg = Bool()
        msg.data = active
        self.ai_active_pub.publish(msg)

    def _publish_motor_stop(self):
        msg = String()
        msg.data = json.dumps({'left': 0, 'right': 0, 'mode': 0})
        self.motor_cmd_pub.publish(msg)

    def _publish_mode_state(self):
        msg = String()
        msg.data = self.mode
        self.mode_state_pub.publish(msg)
        self.get_logger().info(f'모드 전환: {self.mode}')

    def destroy_node(self):
        self._stop_tier2()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AriaControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
