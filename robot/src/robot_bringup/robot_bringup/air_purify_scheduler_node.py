#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String

class AirPurifySchedulerNode(Node):
    def __init__(self):
        super().__init__('air_purify_scheduler_node')

        # .env 파일 로드 및 환경변수 파싱
        dotenv_path = Path("/srv/aria/users/hs/aria/.env")
        load_dotenv(dotenv_path=dotenv_path, override=True)
        load_dotenv(dotenv_path=dotenv_path)
        self.get_logger().info(f"🔍 탐색한 .env 경로: {dotenv_path}")
        self.get_logger().info(f"🔍 파일 존재 여부: {dotenv_path.exists()}")

        # 파라미터 선언 (로컬이 아닌 AWS 클라우드 API 주소를 기본값으로 설정)
        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter(
            'server_url', 
            os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com')
        )
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))

        # 4단계 분류용 임계값
        self.declare_parameter('threshold_good', 30.0) # 이 미만은 '좋음'
        self.declare_parameter('threshold_bad', 70.0)  # 이 이상은 '나쁨' (청정 대상)

        # pm25/voc 원본값 -> 0~100점 점수 변환 기준값
        self.declare_parameter('pm25_max', 100.0)
        self.declare_parameter('voc_max', 150.0)

        self._robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self._server_url = str(self.get_parameter('server_url').get_parameter_value().string_value).rstrip('/')
        self._auth_token = self.get_parameter('auth_token').get_parameter_value().string_value
        self._threshold_good = self.get_parameter('threshold_good').get_parameter_value().double_value
        self._threshold_bad = self.get_parameter('threshold_bad').get_parameter_value().double_value
        self._pm25_max = self.get_parameter('pm25_max').get_parameter_value().double_value
        self._voc_max = self.get_parameter('voc_max').get_parameter_value().double_value

        # Nav2 Action Client
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 내부 상태 및 큐 관리
        self.state = 'IDLE' # IDLE, PATROLLING, PURIFYING
        self.patrol_queue = []
        self.purify_queue = []
        self.measured_scores = [] 
        self.current_target_zone = None

        # 실시간 공기질 데이터 구독
        self._current_air_score = -1.0
        self._air_quality_sub = self.create_subscription(
            String, '/esp32/air_raw', self.air_quality_callback, 10
        )

        # MQTT 노드로 데이터를 송신하기 위한 ROS2 Publisher
        self.air_quality_pub = self.create_publisher(String, '/aria/zone_air_quality', 10)

        # aria_controller_node 트리거 구독
        self._start_cycle_sub = self.create_subscription(
            Empty, '/aria/start_purify_cycle', self.on_start_cycle_trigger, 10
        )

        # 중단 토픽 구독
        self._abort_sub = self.create_subscription(
            Empty, '/aria/purify_abort', self.on_purify_abort, 10
        )

        # 순회 완료 알림 퍼블리셔
        self.purify_done_pub = self.create_publisher(Empty, '/aria/purify_cycle_done', 10)

        self.get_logger().info('🤖 AI Purify Scheduler가 성공적으로 시작되었습니다.')
        self.get_logger().info(f'설정된 로봇 ID: {self._robot_id}, 서버 주소: {self._server_url}')

    def air_quality_callback(self, msg):
        """ESP32 원본 pm25/voc 값을 0~100점 공기질 점수로 변환한다."""
        try:
            data = json.loads(msg.data)
            pm25 = float(data.get('pm25', 0.0))
            voc = float(data.get('voc', 0.0))

            score = max(pm25 / self._pm25_max, voc / self._voc_max) * 100.0
            self._current_air_score = min(100.0, max(0.0, score))
        except Exception as e:
            self.get_logger().warn(f'⚠️ 공기질 원본 데이터 파싱 실패: {e}')

    def on_start_cycle_trigger(self, msg):
        self.start_ai_cycle()

    def on_purify_abort(self, msg):
        if self.state == 'IDLE':
            return
        self.get_logger().warn('🛑 순회/정화가 외부 요청으로 중단되었습니다. IDLE로 전환합니다.')
        self.state = 'IDLE'
        self.patrol_queue.clear()
        self.purify_queue.clear()
        self.measured_scores.clear()
        self.current_target_zone = None

    def start_ai_cycle(self):
        if self.state != 'IDLE':
            self.get_logger().info('⏳ 이전 주기 작업이 수행 중입니다. 이번 순회를 건너뜁니다.')
            return

        self.get_logger().info('🔄 [순회 트리거 수신] 구역 순회 및 공기질 측정을 개시합니다.')
        
        try:
            url = f'{self._server_url}/robots/{self._robot_id}/zones'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token

            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                self.get_logger().error(f'❌ API 서버 응답 오류 (코드: {response.status_code}, 내용: {response.text})')
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
            self.get_logger().error(f'❌ 구역 데이터 동기화 중 예외 발생: {e}')
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
        self.get_logger().info(f"🚀 [{mode_str}] 목적지 '{zone.get('name')}'(으)로 주행을 시작합니다.")

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ 네비게이션 목표가 거부되었습니다. 다음 구역으로 진행합니다.')
            self.visit_next_zone()
            return
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        zone_name = self.current_target_zone.get('name')

        if self.state == 'PATROLLING':
            measured_score = self._current_air_score
            self.measured_scores.append({'zone': self.current_target_zone, 'score': measured_score})
            self.visit_next_zone()

        elif self.state == 'PURIFYING':
            self.get_logger().info(f"✨ '{zone_name}' 집중 정화 완료. 최종 개선 데이터를 업로드합니다.")
            self.upload_air_quality_to_cloud(self.current_target_zone)
            self.visit_next_zone()

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
                "air_grade": grade
            }
            
            msg = String()
            msg.data = json.dumps(payload)
            self.air_quality_pub.publish(msg)
            
            self.get_logger().info(f"📤 [{grade}] '{zone.get('name')}' 최종 결과({score_int}점) MQTT 노드로 송신 완료")
            
        except Exception as e:
            self.get_logger().error(f'❌ MQTT 토픽 송신 중 에러 발생: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AirPurifySchedulerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
