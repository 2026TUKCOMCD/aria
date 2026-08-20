import os
import csv
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        # 1. 환경 변수 및 API 파라미터 세팅 (기존 코드 재활용)
        dotenv_path = Path("/srv/aria/users/hs/aria/.env")
        load_dotenv(dotenv_path=dotenv_path, override=True)
        
        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter('server_url', os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com'))
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))

        self._robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self._server_url = str(self.get_parameter('server_url').get_parameter_value().string_value).rstrip('/')
        self._auth_token = self.get_parameter('auth_token').get_parameter_value().string_value

        # 2. 데이터 저장을 위한 CSV 설정
        self.save_dir = os.path.expanduser('~/aria_ai_system/data')
        self.csv_path = os.path.join(self.save_dir, 'aria_daily_log.csv')
        os.makedirs(self.save_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['hour', 'minute', 'day_of_week', 'zone_name', 'pm25'])

        # 3. 공기질 데이터 구독 (기존 코드 재활용)
        self.current_pm25 = 0.0
        self._air_quality_sub = self.create_subscription(String, '/esp32/air_raw', self.air_quality_callback, 10)

        # 4. Nav2 Action Client (기존 코드 재활용)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 순회 관리를 위한 큐
        self.patrol_queue = []
        self.current_target_zone = None
        self.is_patrolling = False

        # 5. 2시간(7200초)마다 순회 수집을 시작하는 타이머
        # 테스트 시에는 이 값을 60(1분) 등으로 줄여서 확인하세요.
        self.timer_period = 7200.0  
        self.timer = self.create_timer(self.timer_period, self.start_patrol_cycle)

        self.get_logger().info('🤖 AI 데이터 수집기(Data Collector)가 시작되었습니다.')
        self.get_logger().info(f'기록 경로: {self.csv_path}')
        
        # 노드 켜지자마자 바로 1회 수집 시작
        self.start_patrol_cycle()

    def air_quality_callback(self, msg):
        """ESP32 파서가 보내주는 JSON에서 PM2.5 값만 추출"""
        try:
            data = json.loads(msg.data)
            self.current_pm25 = float(data.get('pm25', 0.0))
        except Exception as e:
            pass # 파싱 실패 시 무시

    def start_patrol_cycle(self):
        """API 서버에서 구역을 가져와 순회 큐에 넣고 출발"""
        if self.is_patrolling:
            self.get_logger().info('⏳ 아직 이전 순회가 끝나지 않았습니다.')
            return

        self.get_logger().info('🔄 [데이터 수집] 2시간 주기 구역 순회를 시작합니다.')
        try:
            url = f'{self._server_url}/robots/{self._robot_id}/zones'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token

            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                zones = response.json().get('zones', [])
                if zones:
                    self.is_patrolling = True
                    self.patrol_queue = zones.copy()
                    self.visit_next_zone()
                else:
                    self.get_logger().warn('⚠️ 등록된 구역 정보가 없습니다.')
        except Exception as e:
            self.get_logger().error(f'❌ API 구역 데이터 동기화 실패: {e}')

    def visit_next_zone(self):
        """큐에서 다음 구역을 꺼내 이동 명령 전송"""
        if not self.patrol_queue:
            self.is_patrolling = False
            self.get_logger().info('✅ 이번 주기(2시간)의 모든 구역 데이터 수집이 완료되었습니다.')
            return

        self.current_target_zone = self.patrol_queue.pop(0)
        zone_name = self.current_target_zone.get('name')
        center = self.current_target_zone.get('center', {})
        x, y = center.get('x'), center.get('y')

        if x is None or y is None:
            self.visit_next_zone()
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"🚀 측정 목적지 '{zone_name}'(으)로 이동합니다.")
        
        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ 이동 목표가 거부되었습니다. 다음 구역으로 넘어갑니다.')
            self.visit_next_zone()
            return

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """목표 구역에 도착하면 CSV에 데이터를 기록하고 다음 구역으로 이동"""
        zone_name = self.current_target_zone.get('name')
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()

        # 도착 직후의 미세먼지 수치(self.current_pm25) 기록
        try:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([hour, minute, day_of_week, zone_name, self.current_pm25])
            self.get_logger().info(f"📝 데이터 기록 완료! [{zone_name}] PM2.5: {self.current_pm25:.1f}")
        except Exception as e:
            self.get_logger().error(f"데이터 기록 실패: {e}")

        # 기록 완료 후 남은 다음 구역으로 이동
        self.visit_next_zone()

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
