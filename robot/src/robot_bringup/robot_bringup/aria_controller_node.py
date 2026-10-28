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

    "모든 노드를 컨트롤하는 메인 통합 노드" 그 자체 — AI(순회/예측청정)
    노드들은 my_robot_ai_control(data_collector_node, ai_cleaner_node)로
    분리되어 여기엔 pandas/scikit-learn 등 무거운 AI 의존성이 없다.

    모드: STANDBY(대기) / AI(순회+예측청정) / MAPPING(매핑) / BASIC(기본모드)
    초기 상태는 항상 STANDBY — 부팅 즉시 순회를 시작하지 않는다.

    이 노드가 유일하게 소유하는 것:
      - 수동이동/기본모드/매핑귀환용 NavigateToPose ActionClient
        (순회/집중정화 전용 ActionClient는 data_collector_node,
         예측청정 이동 전용 ActionClient는 ai_cleaner_node가 각자 소유 —
         활성 구간이 /aria/ai_active와 /aria/start_purify_cycle↔
         /aria/purify_cycle_done 핸드셰이크로 서로 겹치지 않도록 보장된다)
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
        # 순회+집중정화 전체를 기다리는 최대 시간. 집중정화가 구역당 최대 10분이므로
        # 600초로는 부족하다(넘기면 순회 중에 예측청정이 켜져 두 노드가 충돌).
        self.declare_parameter('purify_cycle_timeout', 3600.0)

        # 수면 시간 기본값. 웹앱에서 저장한 값이 MQTT(/aria/sleep_schedule)로 오면 덮어쓴다.
        self.declare_parameter('sleep_time', '23:00')
        self.declare_parameter('wake_time', '07:00')
        self.declare_parameter('sleep_enabled', True)
        self.declare_parameter('sleep_schedule_file', os.path.expanduser('~/aria_ai_system/sleep_schedule.json'))
        # 취침 모드 진입 직전 모드(기상 시 복귀할 모드)를 저장하는 파일.
        # 수면 중 재부팅돼도 기상 시각에 원래 모드로 돌아가기 위해 파일로 둔다.
        self.declare_parameter('sleep_state_file', os.path.expanduser('~/aria_ai_system/sleep_state.json'))

        # 기본모드 점수 계산 (data_collector_node와 동일 공식/기본값)
        self.declare_parameter('threshold_good', 30.0)
        self.declare_parameter('pm25_max', 100.0)
        self.declare_parameter('voc_max', 150.0)
        # 기본모드 정화: 도착 후 팬을 켜고 공기질이 양호해질 때까지 머문다.
        # 최소 시간은 센서 반응 지연(PM 센서 2초 주기·실내 확산) 때문에 둔다(data_collector와 동일).
        # 최대 시간은 센서 고장이나 오염원이 계속 있을 때 끝없이 머무는 것을 막는다.
        self.declare_parameter('basic_fan_speed', 255)
        self.declare_parameter('basic_purify_min_sec', 60.0)
        self.declare_parameter('basic_purify_max_sec', 1800.0)

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
        self.purify_cycle_timeout = float(self.get_parameter('purify_cycle_timeout').value)
        self.sleep_schedule_file = self.get_parameter('sleep_schedule_file').value
        self.sleep_state_file = self.get_parameter('sleep_state_file').value
        self._sleep_time = self._parse_hhmm(self.get_parameter('sleep_time').value, dt_time(23, 0))
        self._wake_time = self._parse_hhmm(self.get_parameter('wake_time').value, dt_time(7, 0))
        self._sleep_enabled = bool(self.get_parameter('sleep_enabled').value)
        self._load_sleep_schedule_file()  # 재부팅·오프라인 대비 마지막 저장값 복원
        self.threshold_good = float(self.get_parameter('threshold_good').value)
        self._pm25_max = float(self.get_parameter('pm25_max').value)
        self._voc_max = float(self.get_parameter('voc_max').value)
        self.basic_fan_speed = int(self.get_parameter('basic_fan_speed').value)
        self.basic_purify_min_sec = float(self.get_parameter('basic_purify_min_sec').value)
        self.basic_purify_max_sec = float(self.get_parameter('basic_purify_max_sec').value)
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
        # 기본모드 정화 세션 번호. 새 이동 명령·취소·대기 전환 때마다 올려서,
        # 이전 정화 루프가 스스로 끝나고 새 루프가 켠 팬을 끄지 않게 한다.
        self._basic_session = 0
        self._basic_lock = threading.Lock()
        self._resume_mode = 'STANDBY'   # 취침 모드에서 기상할 때 돌아갈 모드
        self._boot_resume_mode = None   # 수면 중 재부팅 시 파일에서 복원한 복귀 모드
        self._sleeping = False         # 수면 시간이라 AI를 멈춘 상태인가
        self._ai_active_state = False   # 마지막으로 발행한 /aria/ai_active 값
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
        # ai_cleaner가 예측에 없던 오염을 감지하면 즉시 순회를 요청한다.
        self.create_subscription(Empty, '/aria/request_patrol', self._on_patrol_request, 10)
        self.create_subscription(String, '/aria/sleep_schedule', self._on_sleep_schedule, 10)

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
        self.fan_cmd_pub = self.create_publisher(String, '/esp32/fan_cmd', 10)
        self.sleep_state_pub = self.create_publisher(Bool, '/aria/sleep_state', 10)
        # 매핑 완료/중단 등 사용자에게 알릴 이벤트 → aria_mqtt_node가 웹 알림으로 중계
        self.robot_event_pub = self.create_publisher(String, '/aria/robot_event', 10)

        # AI모드 4시간 재실행 감시 (1초 주기)
        self.create_timer(1.0, self._ai_mode_tick)

        self._load_sleep_state_file()  # 수면 중 재부팅 대비: 저장된 복귀 모드 읽기

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

        if self.mode == 'SLEEP':
            # 취침 중에는 바로 실행하지 않고 '기상 후 돌아갈 모드'만 바꾼다.
            target = self._normalize_mode(mode)
            if target in ('STANDBY', 'AI', 'BASIC'):
                self._resume_mode = target
                self._save_sleep_state()
                self.get_logger().info(f'취침 중 모드 요청: 기상 후 {target} 모드로 돌아갑니다.')
            else:
                self.get_logger().warn(f'취침 중에는 {mode!r} 요청을 실행할 수 없습니다.')
            return

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
        self._next_basic_session()  # 진행 중인 기본모드 정화 루프 종료
        self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._set_ai_active(False)
        self._publish_motor_stop()
        self._publish_fan(0)  # 기본모드 정화 중 '청정 중지'를 눌러도 팬이 바로 꺼지도록

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
        threading.Thread(target=self._ai_mode_worker, daemon=True).start()

    def _ai_mode_worker(self):
        if self._is_night_time():
            # 수면 중에 AI모드가 켜지면 순회도 예측도 하지 않는다.
            # 기상 시각에 예측청정만 켜고, 순회는 그때부터 4시간 주기로 돈다.
            self.get_logger().info('수면 시간이므로 기상 시각까지 대기합니다.')
            self._purify_retrigger_in_progress = False
            return
        finished = self._run_purify_cycle_and_wait()
        if self.mode != 'AI':
            return  # 대기중 다른 모드로 전환됨
        if finished and not self._sleeping:
            self._return_to_dock()
            if self.mode != 'AI':
                return
        self._ai_purify_timer_start = time.time()
        self._set_ai_active(True)

    def _return_to_dock(self) -> bool:
        """순회·집중정화가 끝난 뒤 충전기(도킹) 위치로 복귀한다.

        이전엔 순회가 끝나면 마지막 정화 구역에 그대로 멈춰 있었다
        (ai_cleaner는 자기 위치를 'home'으로 알고 있어 복귀시키지도 않았다).
        도착하면 aria_mqtt_node가 target_type=DOCK을 보고
        '충전기 위치로 이동을 완료했습니다.'를 웹 로그로 보낸다.
        """
        dock = self._get_dock_pose()
        if dock is None:
            self.get_logger().warn('저장된 충전기 위치가 없어 복귀를 건너뜁니다.')
            return False
        self.get_logger().info('🏠 순회 종료 — 충전기 위치로 복귀합니다.')
        return self._navigate_to(
            dock['x'], dock['y'], dock.get('theta', 0.0),
            timeout=self.nav_timeout_sec,
            metadata={'target_type': 'DOCK', 'zone_name': '충전기 위치'},
        )

    def _on_patrol_request(self, msg: Empty):
        """ai_cleaner가 '예측에 없던 오염'을 감지했을 때 보내는 순회 요청.

        4시간 주기 순회와 같은 경로(_rerun_purify_cycle)를 즉시 실행한다:
        탐색(전 구역 측정) → 구역별 점수 → 나쁨 구역 우선순위 정화 → 충전기 복귀.
        """
        if self.mode != 'AI' or self._sleeping:
            return
        if self._ai_purify_timer_start is None or self._purify_retrigger_in_progress:
            self.get_logger().info('이미 순회가 진행 중이라 오염 감지 순회 요청을 무시합니다.')
            return
        self.get_logger().info('🚨 오염 감지 — 즉시 탐색 순회를 시작합니다.')
        self._start_rerun_purify_cycle()

    def _start_rerun_purify_cycle(self):
        self._purify_retrigger_in_progress = True
        self._set_ai_active(False)
        threading.Thread(target=self._rerun_purify_cycle, daemon=True).start()

    def _run_purify_cycle_and_wait(self):
        self._purify_done_event.clear()
        self.purify_trigger_pub.publish(Empty())
        finished = self._purify_done_event.wait(timeout=self.purify_cycle_timeout)
        if not finished:
            # 끝나지 않은 순회를 그대로 두면 예측청정과 이동 목표가 겹친다. 명시적으로 중단.
            self.get_logger().warn(f'순회가 {self.purify_cycle_timeout:.0f}s 안에 끝나지 않아 중단합니다.')
            self.purify_abort_pub.publish(Empty())
        return finished

    def _ai_mode_tick(self):
        self._update_sleep_state()
        if self.mode != 'AI' or self._sleeping:
            return
        if self._ai_purify_timer_start is None:
            return  # 최초 순회가 아직 안 끝남
        if self._purify_retrigger_in_progress:
            return
        elapsed = time.time() - self._ai_purify_timer_start
        if elapsed < self.purify_interval:
            return
        self._start_rerun_purify_cycle()

    def _rerun_purify_cycle(self):
        try:
            finished = self._run_purify_cycle_and_wait()
            if self.mode != 'AI' or self._sleeping:
                return
            if finished:
                self._return_to_dock()
                if self.mode != 'AI' or self._sleeping:
                    return
            self._ai_purify_timer_start = time.time()
            self._set_ai_active(True)
        finally:
            self._purify_retrigger_in_progress = False

    def _on_purify_cycle_done(self, msg: Empty):
        self._purify_done_event.set()

    # ══════════════════════════════════════════════════
    # 수면(절전) 시간 — 취침 모드
    # ══════════════════════════════════════════════════
    #
    # 수면 시각이 되면 현재 모드(STANDBY/AI/BASIC)를 '복귀 모드'로 파일에 저장하고
    # 모든 동작(이동·순회·예측·팬·모터)을 멈춘 SLEEP 모드로 들어간다.
    # 기상 시각이 되면 저장된 모드로 되돌아간다.
    #   - AI    : 예측 청정만 즉시 재개, 다음 순회는 그때부터 4시간 뒤(기상 직후 순회 없음)
    #   - BASIC : 기본모드 대기 상태(진행 중이던 정화 작업은 다시 시작하지 않음)
    #   - STANDBY: 대기
    # 매핑 중에 수면 시각이 되면 매핑이 끝난 뒤에 취침 모드로 들어간다.
    # 취침 중 웹에서 온 모드 명령은 바로 실행하지 않고 '복귀 모드'만 바꾼다.

    @staticmethod
    def _parse_hhmm(value, default: dt_time) -> dt_time:
        try:
            h, m = str(value).strip().split(':')[:2]
            return dt_time(int(h), int(m))
        except Exception:
            return default

    def _is_night_time(self) -> bool:
        if not self._sleep_enabled:
            return False
        now = datetime.now().time()
        start, end = self._sleep_time, self._wake_time
        if start == end:
            return False
        if start < end:
            return start <= now < end
        return now >= start or now < end

    def _on_sleep_schedule(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'수면 스케줄 파싱 실패: {e}')
            return
        self._apply_sleep_schedule(data)
        try:
            os.makedirs(os.path.dirname(self.sleep_schedule_file), exist_ok=True)
            with open(self.sleep_schedule_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.get_logger().warn(f'수면 스케줄 저장 실패: {e}')

    def _load_sleep_schedule_file(self):
        try:
            with open(self.sleep_schedule_file) as f:
                self._apply_sleep_schedule(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warn(f'수면 스케줄 파일 읽기 실패: {e}')

    def _apply_sleep_schedule(self, data: dict):
        self._sleep_time = self._parse_hhmm(data.get('sleep_time'), self._sleep_time)
        self._wake_time = self._parse_hhmm(data.get('wake_time'), self._wake_time)
        self._sleep_enabled = bool(data.get('enabled', self._sleep_enabled))
        self.get_logger().info(
            f'수면 스케줄 적용: {self._sleep_time:%H:%M} ~ {self._wake_time:%H:%M} '
            f'(사용={self._sleep_enabled})'
        )

    def _update_sleep_state(self):
        night = self._is_night_time()
        if night == self._sleeping:
            return
        if night:
            if self.mode == 'MAPPING':
                return  # 매핑이 끝나 대기모드가 되면 다음 틱에 취침 모드로 들어간다
            resume = self._boot_resume_mode or self.mode
            self._boot_resume_mode = None
            self._enter_sleep(resume)
        else:
            self._exit_sleep()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        mode = str(mode).strip().upper()
        if mode in ('STANDBY', '대기', '대기모드', 'WAIT'):
            return 'STANDBY'
        if mode in ('AI', 'AI_MODE', 'AI모드', 'AUTO'):
            return 'AI'
        if mode in ('BASIC', '기본모드', 'MANUAL'):
            return 'BASIC'
        if mode in ('MAPPING', '매핑', 'SLAM'):
            return 'MAPPING'
        return mode

    def _enter_sleep(self, resume_mode: str):
        if resume_mode not in ('STANDBY', 'AI', 'BASIC'):
            resume_mode = 'STANDBY'
        self._sleeping = True
        self._resume_mode = resume_mode
        self._save_sleep_state()
        self.get_logger().info(f'🌙 취침 모드 시작 (기상 후 {resume_mode} 모드로 복귀)')

        # 진행 중인 모든 동작 정지 (STANDBY 진입과 같은 정리 + 대기 중인 순회 스레드 해제)
        self.mode = 'SLEEP'
        self._next_basic_session()
        self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._purify_done_event.set()
        self._set_ai_active(False, force=True)
        self._publish_motor_stop()
        self._publish_fan(0)

        self.sleep_state_pub.publish(Bool(data=True))
        self._publish_mode_state()
        self._publish_robot_event('SLEEP_START', '취침 모드 시작')

    def _exit_sleep(self):
        resume_mode = self._resume_mode
        self._sleeping = False
        self._clear_sleep_state()
        self.sleep_state_pub.publish(Bool(data=False))
        self.get_logger().info(f'☀️ 기상 시각: 취침 전 모드({resume_mode})로 돌아갑니다.')

        if resume_mode == 'AI':
            self.mode = 'AI'
            self._purify_retrigger_in_progress = False
            self._ai_purify_timer_start = time.time()
            self._publish_mode_state()
            self._set_ai_active(True)
        elif resume_mode == 'BASIC':
            self.mode = 'BASIC'
            self._publish_mode_state()
        else:
            self.mode = 'STANDBY'
            self._publish_mode_state()

    def _save_sleep_state(self):
        try:
            os.makedirs(os.path.dirname(self.sleep_state_file), exist_ok=True)
            tmp = self.sleep_state_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({
                    'sleeping': True,
                    'resume_mode': self._resume_mode,
                    'saved_at': datetime.now().isoformat(timespec='seconds'),
                }, f)
            os.replace(tmp, self.sleep_state_file)
        except Exception as e:
            self.get_logger().warn(f'취침 상태 저장 실패: {e}')

    def _clear_sleep_state(self):
        try:
            os.remove(self.sleep_state_file)
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warn(f'취침 상태 파일 삭제 실패: {e}')

    def _load_sleep_state_file(self):
        """부팅 시 호출. 수면 중 재부팅이면 저장된 복귀 모드를 이어받는다.

        이미 기상 시각이 지났으면 파일만 지우고 대기모드로 시작한다
        (부팅 직후 로봇이 저절로 AI 모드에 들어가지 않게 하는 기존 원칙 유지).
        """
        try:
            with open(self.sleep_state_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            self.get_logger().warn(f'취침 상태 파일 읽기 실패: {e}')
            return
        if data.get('sleeping') and self._is_night_time():
            self._boot_resume_mode = self._normalize_mode(data.get('resume_mode', 'STANDBY'))
            self.get_logger().info(f'수면 중 재부팅 — 기상 후 {self._boot_resume_mode} 모드로 복귀합니다.')
        else:
            self._clear_sleep_state()

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
                self._publish_robot_event('MAPPING_FAILED', '맵 생성이 제한 시간 안에 끝나지 않았습니다.')
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
                self._publish_robot_event('MAPPING_COMPLETED', '맵 생성이 완료되었습니다. 맵을 새로고침해 주세요.')
            else:
                self._publish_robot_event('MAPPING_FAILED', '맵 저장에 실패했습니다.')

            self._swap_tier2('LOCALIZE', map_yaml=f'{map_path}.yaml' if map_path else None)
            self._call_trigger_service_sync(self.reinit_client)

        except Exception as e:
            self.get_logger().error(f'매핑 시퀀스 오류: {e}')
            self._publish_robot_event('MAPPING_FAILED', f'맵 생성 중 오류가 발생했습니다: {e}')
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
        # 웹앱은 'MODE=MANUAL'(Shadow)과 '구역 이동'(cmd/nav)을 동시에 보낸다.
        # 이동 명령이 먼저 도착한 경우 여기서 취소하면 로봇이 출발하지 않으므로,
        # 기본모드 명령 자체에 목표가 담겨 있을 때만 기존 이동을 취소한다.
        if any(k in payload for k in ('x', 'y', 'zone_id')):
            self._cancel_own_nav_goal()
        self.purify_abort_pub.publish(Empty())
        self._set_ai_active(False)
        self.mode = 'BASIC'
        self._publish_mode_state()
        threading.Thread(target=self._basic_mode_worker, args=(payload,), daemon=True).start()

    def _basic_mode_worker(self, payload: dict):
        target = self._resolve_basic_target(payload)
        if target is None or target.get('x') is None or target.get('y') is None:
            # 웹앱은 기본모드 전환(MODE=MANUAL)과 구역 이동(cmd/nav MOVE_TO)을 따로 보낸다.
            # 좌표 없는 모드 명령은 '기본모드로 바꿔라'만 뜻하고, 실제 이동·정화는
            # 뒤이어 오는(또는 먼저 온) MOVE_TO를 _manual_move_worker가 맡는다.
            self.get_logger().info('기본모드 진입 — 구역 이동 명령이 오면 이동 후 정화합니다.')
            return

        session = self._next_basic_session()
        name = target.get('name', '기본모드 목표')
        arrived = self._navigate_to(
            target['x'], target['y'], target.get('theta', 0.0),
            timeout=self.nav_timeout_sec,
            metadata={'target_type': 'BASIC', 'zone_name': name},
        )
        if not self._basic_session_alive(session):
            return  # 이동 중에 다른 명령이 들어옴
        if not arrived:
            self.get_logger().warn('기본모드 목표 이동 실패. 대기모드로 복귀합니다.')
            self._enter_standby()
            return

        self._basic_purify_here(session, name)

    # ── 기본모드 정화 (도착 후 팬 가동 → 공기질 양호까지 체류) ──

    def _next_basic_session(self) -> int:
        with self._basic_lock:
            self._basic_session += 1
            return self._basic_session

    def _basic_session_alive(self, session: int) -> bool:
        return self.mode == 'BASIC' and session == self._basic_session

    def _basic_purify_here(self, session: int, zone_name: str):
        """기본모드 목표에 도착한 뒤 호출. 팬을 켜고 공기질이 양호해질 때까지 머문다.

        종료 조건
          - 최소 시간(basic_purify_min_sec)이 지났고 종합 점수가 threshold_good 이하
          - 최대 시간(basic_purify_max_sec) 도달
        중단 조건(팬만 끄고 조용히 끝냄)
          - 모드가 BASIC이 아니게 됨(대기·AI·매핑 전환)
          - 새 이동 명령·이동 취소로 세션이 바뀜(새 세션이 팬을 소유)
        """
        self._current_air_score = -1.0  # 도착 전(이동 중) 값으로 판단하지 않도록
        started = time.time()
        self._publish_fan(self.basic_fan_speed)
        self._publish_robot_event('PURIFY_START', '공기청정을 시작합니다.')
        self.get_logger().info(
            f"🌀 기본모드 '{zone_name}' 정화 시작 (최소 {self.basic_purify_min_sec:.0f}s / "
            f"최대 {self.basic_purify_max_sec:.0f}s)"
        )

        reason = None
        try:
            while self._basic_session_alive(session):
                elapsed = time.time() - started
                score = self._current_air_score
                if elapsed >= self.basic_purify_min_sec and 0.0 <= score <= self.threshold_good:
                    reason = '공기질 양호'
                    break
                if elapsed >= self.basic_purify_max_sec:
                    reason = '최대 시간 도달'
                    break
                time.sleep(1.0)
        finally:
            # 세션이 바뀌었으면 새 세션이 팬을 쓰고 있으므로 건드리지 않는다.
            if session == self._basic_session:
                self._publish_fan(0)

        if reason is None:
            self.get_logger().info(f"기본모드 '{zone_name}' 정화 중단(다른 명령).")
            return

        self.get_logger().info(f"✨ 기본모드 '{zone_name}' 정화 완료({reason}). 대기모드로 복귀합니다.")
        # 기본모드는 사용자가 '청정 시작'을 누른 한 번의 작업이므로, 끝나면 웹의 실행 상태도
        # 풀어야 한다. 웹은 PURIFYING_DONE을 받으면 실행 중 표시를 끄고 완료 팝업을 띄운다.
        # (AI모드의 반복 정화는 PURIFY_DONE을 써서 이 동작을 일으키지 않는다.)
        self._publish_robot_event('PURIFYING_DONE', '청정이 완료되었습니다.')
        if self.mode == 'BASIC':
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
        if self.mode == 'SLEEP':
            self.get_logger().warn('취침 모드 중에는 이동 명령을 무시합니다.')
            return

        cmd_type = str(payload.get('type', 'MOVE_TO')).upper()
        if cmd_type in ('CANCEL', 'STOP'):
            self._next_basic_session()  # 기본모드 정화 중이면 루프 종료 + 팬 끄기
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

        # 새 이동이 시작되면 이전 기본모드 정화는 끝낸다(새 목적지에서 다시 시작).
        session = self._next_basic_session()
        threading.Thread(
            target=self._manual_move_worker, args=(float(x), float(y), theta, metadata, session), daemon=True
        ).start()

    def _manual_move_worker(self, x: float, y: float, theta: float, metadata: dict, session: int):
        arrived = self._navigate_to(x, y, theta, timeout=self.nav_timeout_sec, metadata=metadata)
        target_type = str(metadata.get('target_type', '')).upper()
        if arrived and target_type == 'DOCK_SETUP':
            self._call_trigger_service_sync(self.save_dock_client)
            return
        if target_type == 'DOCK':
            return  # 충전기 위치로 이동은 정화하지 않는다

        # 웹앱 기본모드: MODE=MANUAL(→BASIC)과 MOVE_TO가 따로 오므로, 도착 시점에
        # 기본모드이면 그 자리에서 정화를 시작한다. 모드 명령이 이동 명령보다 늦게 와도
        # 이동하는 동안 이미 도착해 있으므로 순서와 상관없이 동작한다.
        # 대기모드에서의 이동(충전기 위치로 이동 등)은 정화하지 않는다.
        if not self._basic_session_alive(session):
            return
        if not arrived:
            self.get_logger().warn('기본모드 목표 이동 실패. 대기모드로 복귀합니다.')
            self._enter_standby()
            return
        zone_name = metadata.get('zone_name') or metadata.get('target_name') or '지정 위치'
        self._basic_purify_here(session, zone_name)

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
        timed_out = [False]

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
            if accepted and timed_out[0]:
                # 이미 실패로 처리된 뒤 늦게 수락된 목표 — 바로 취소한다.
                goal_handle.cancel_goal_async()
                return
            with self._nav_goal_lock:
                self._current_goal_handle = goal_handle if accepted else None
            if not accepted:
                arrived.set()
                return
            goal_handle.get_result_async().add_done_callback(result_cb)

        self._publish_nav_event('NAVIGATION_REQUESTED', metadata)
        self.nav_client.send_goal_async(goal).add_done_callback(response_cb)
        finished = arrived.wait(timeout=timeout)

        with self._nav_goal_lock:
            goal_handle = self._current_goal_handle
            self._current_goal_handle = None

        if not finished:
            # 타임아웃: 실패로 보고만 하고 목표를 그대로 두면 로봇이 계속 이동한다.
            timed_out[0] = True
            self.get_logger().warn(f'이동 제한 시간({timeout:.0f}s) 초과 — Nav2 목표를 취소합니다.')
            if goal_handle is not None:
                goal_handle.cancel_goal_async()
            extra = {'reason': '이동 제한 시간 초과'}
            self._publish_nav_event('NAVIGATION_FAILED', metadata, **extra)
            return False

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
            cmd = ['ros2', 'launch', 'my_robot_bringup', 'localize.launch.py']
            if map_yaml:
                cmd.append(f'map_yaml:={map_yaml}')
        else:  # 'MAP'
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

    def _set_ai_active(self, active: bool, force: bool = False):
        # 수면 중에는 어떤 경로로도 예측청정을 켜지 않는다(단일 관문).
        if active and self._sleeping and not force:
            self.get_logger().info('수면 시간이므로 예측청정 활성화를 보류합니다.')
            active = False
        self._ai_active_state = active
        msg = Bool()
        msg.data = active
        self.ai_active_pub.publish(msg)

    def _publish_fan(self, speed: int):
        msg = String()
        msg.data = json.dumps({'fan': int(speed)})
        self.fan_cmd_pub.publish(msg)

    def _publish_motor_stop(self):
        msg = String()
        msg.data = json.dumps({'left': 0, 'right': 0, 'mode': 0})
        self.motor_cmd_pub.publish(msg)

    def _publish_mode_state(self):
        msg = String()
        msg.data = self.mode
        self.mode_state_pub.publish(msg)
        self.get_logger().info(f'모드 전환: {self.mode}')

    def _publish_robot_event(self, event_type: str, message: str):
        # aria_mqtt_node._on_robot_event → aria/{id}/event/noti → 웹 알림 + 이벤트 로그
        msg = String()
        msg.data = json.dumps({'type': event_type, 'message': message}, ensure_ascii=False)
        self.robot_event_pub.publish(msg)

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
