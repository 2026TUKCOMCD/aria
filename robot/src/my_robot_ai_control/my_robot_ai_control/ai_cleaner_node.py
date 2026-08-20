import os
import requests
import json
import time
from datetime import datetime
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

class AiCleanerNode(Node):
    def __init__(self):
        super().__init__('ai_cleaner_node')

        dotenv_path = Path("/srv/aria/users/hs/aria/.env")
        load_dotenv(dotenv_path=dotenv_path, override=True)

        self.declare_parameter('robot_id', os.environ.get('ROBOT_ID', '1'))
        self.declare_parameter('server_url', os.environ.get('ARIA_API_URL', 'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com'))
        self.declare_parameter('auth_token', os.environ.get('ARIA_AUTH_TOKEN', ''))

        self._robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self._server_url = str(self.get_parameter('server_url').get_parameter_value().string_value).rstrip('/')
        self._auth_token = self.get_parameter('auth_token').get_parameter_value().string_value

        self.zone_dict = {}
        self.fetch_zone_coordinates()

        self.home_coords = {'x': 0.14, 'y': 0.00}

        model_dir = os.path.expanduser('~/aria_ai_system/models')
        self.model_path = os.path.join(model_dir, 'aria_rf_model.pkl')
        self.model = None
        self.load_ai_model()

        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.fan_pub = self.create_publisher(String, '/esp32/fan_cmd', 10)
        self.air_sub = self.create_subscription(String, '/esp32/air_raw', self.air_callback, 10)

        self.is_moving = False
        self.is_purifying = False
        self.current_target_zone = 'home'
        self.threshold = 0.70
        self.good_threshold = 15.0

        # [핵심 추가] 각 구역별 마지막 청정 완료 시간을 기억하는 딕셔너리
        self.cooldown_dict = {}
        # [핵심 추가] 쿨타임 시간 설정 (30분 = 1800초)
        self.cooldown_duration = 60.0 

        self.timer_period = 10.0
        self.timer = self.create_timer(self.timer_period, self.predict_and_move)

        self.get_logger().info('🤖 AI 예측 청정기(AI Cleaner)가 대기 중입니다.')

    def fetch_zone_coordinates(self):
        try:
            url = f'{self._server_url}/robots/{self._robot_id}/zones'
            headers = {'Accept': 'application/json'}
            if self._auth_token:
                headers['Authorization'] = self._auth_token

            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                zones = response.json().get('zones', [])
                for z in zones:
                    name = z.get('name')
                    center = z.get('center')
                    if name and center:
                        self.zone_dict[name] = center
                self.get_logger().info(f'✅ {len(self.zone_dict)}개의 구역 좌표를 성공적으로 불러왔어.')
        except Exception as e:
            pass

    def load_ai_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.get_logger().info('🧠 AI 예측 모델 로드 완료!')
            except Exception:
                pass

    def send_fan_cmd(self, fan_speed: int):
        msg = String()
        msg.data = json.dumps({"fan": fan_speed})
        self.fan_pub.publish(msg)
        self.get_logger().info(f"💨 팬 가동 상태 변경: {fan_speed}")

    def return_to_home(self):
        if self.is_moving or self.current_target_zone == 'home':
            return
        self.get_logger().info("🏠 임무 완료. 충전 대기소(Home)로 복귀합니다.")
        self.current_target_zone = 'home'
        self.send_navigation_goal('home', custom_coords=self.home_coords)

    def air_callback(self, msg: String):
        if not self.is_purifying:
            return

        try:
            data = json.loads(msg.data)
            pm25 = data.get('pm25', -1.0)

            if 0 <= pm25 < self.good_threshold:
                self.get_logger().info(f"✨ 공기질 회복 완료 (PM2.5: {pm25}). 사전 청정을 조기 종료합니다.")
                
                # [핵심 추가] 현재 청정을 마친 구역에 현재 시간을 기록하여 쿨타임 시작!
                if self.current_target_zone != 'home':
                    self.cooldown_dict[self.current_target_zone] = time.time()
                    
                self.send_fan_cmd(0)
                self.is_purifying = False
                self.return_to_home()
        except Exception:
            pass

    def predict_and_move(self):
        if self.is_moving or self.model is None or not self.zone_dict:
            return

        now = datetime.now()
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()

        try:
            input_df = pd.DataFrame([[hour, minute, day_of_week]], columns=['hour', 'minute', 'day_of_week'])
            probs = self.model.predict_proba(input_df)[0]
            classes = self.model.classes_
            max_prob = max(probs)
            target_zone_name = classes[list(probs).index(max_prob)]

            if max_prob >= self.threshold:
                # [핵심 추가] 쿨타임 검사 로직
                last_clean_time = self.cooldown_dict.get(target_zone_name, 0.0)
                time_passed = time.time() - last_clean_time
                
                if time_passed < self.cooldown_duration:
                    # 아직 쿨타임이 안 지났으면 해당 구역은 깨끗한 것으로 간주하고 무시
                    if self.current_target_zone != 'home' and not self.is_purifying:
                        self.return_to_home()
                    return # 스킵하여 로그가 도배되지 않도록 함

                # 쿨타임이 지났거나 한 번도 안 간 곳이면 정상 출동
                if self.current_target_zone == target_zone_name:
                    pass
                else:
                    self.current_target_zone = target_zone_name
                    self.is_purifying = False
                    self.get_logger().info(f"🚨 예측 분석 [{hour:02d}:{minute:02d}] '{target_zone_name}' 선제 이동 시작 (확률: {max_prob*100:.1f}%)")
                    self.send_fan_cmd(0)
                    self.send_navigation_goal(target_zone_name)
            else:
                if self.current_target_zone != 'home' and not self.is_purifying:
                    self.return_to_home()
        except Exception:
            pass

    def send_navigation_goal(self, zone_name, custom_coords=None):
        if custom_coords:
            target_coords = custom_coords
        else:
            target_coords = self.zone_dict.get(zone_name)

        if not target_coords:
            return

        self.is_moving = True
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target_coords.get('x', 0.0))
        goal_msg.pose.pose.position.y = float(target_coords.get('y', 0.0))
        goal_msg.pose.pose.orientation.w = 1.0

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.is_moving = False
            return
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.is_moving = False
        
        if self.current_target_zone == 'home':
            self.get_logger().info('🔌 충전 대기소 도착! 다음 예측을 대기합니다.')
        else:
            self.get_logger().info('🏁 목적지 도착! 예측 사전 청정을 시작합니다!')
            self.is_purifying = True
            self.send_fan_cmd(255)

def main(args=None):
    rclpy.init(args=args)
    node = AiCleanerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 강제 종료 신호 수신. 시스템을 멈추고 팬을 정지합니다...")
        node.send_fan_cmd(0)
        time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
