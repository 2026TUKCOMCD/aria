import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class PriorityRoutePlanner(Node):
    def __init__(self):
        super().__init__('priority_route_planner')
        
        # 1. YAML 설정값
        self.resolution = 0.02
        self.origin_x = -1.94
        self.origin_y = -0.567
        
        # 2. Nav2 본부에 명령을 내릴 액션 클라이언트 생성
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.get_logger().info("전략가 노드가 실행되었습니다.")

    def send_goal_from_pixel(self, px, py):
        # 3. 좌표 변환 공식 (B5-3) [cite: 130-141]
        # 여기서는 하단 기준 픽셀 좌표라고 가정합니다.
        world_x = (px * self.resolution) + self.origin_x
        world_y = (py * self.resolution) + self.origin_y
        
        self.get_logger().info(f"픽셀({px}, {py}) -> 미터({world_x:.2f}, {world_y:.2f}) 변환 완료")
        
        # 4. Nav2에게 목적지 전송
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = world_x
        goal_msg.pose.pose.position.y = world_y
        goal_msg.pose.pose.orientation.w = 1.0 # 정면 바라보기
        
        self._action_client.wait_for_server()
        self._action_client.send_goal_async(goal_msg)
        self.get_logger().info("로봇에게 주행 명령을 전달했습니다!")

def main(args=None):
    rclpy.init(args=args)
    planner = PriorityRoutePlanner()
    
    # 5. [데모용 하드코딩] 거실 중앙이라고 생각되는 픽셀 좌표를 넣어보세요 (예: 100, 150)
    # 실제 지도 이미지(pgm)를 보고 로봇이 갈만한 빈 공간의 좌표를 찍어보아야 합니다.
    planner.send_goal_from_pixel(150.0, 150.0)
    
    rclpy.spin(planner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
