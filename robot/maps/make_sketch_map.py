#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from PIL import Image  # PGM 파일 생성을 위해 추가

class StaticMapNode(Node):
    def __init__(self):
        super().__init__('aria_static_map_node')
        
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)
        
        # --- [추가] 맵 데이터 생성 및 파일 저장 (노드 시작 시 1회만 실행) ---
        self.map_msg = self.create_map_data()
        self.save_map_to_files(self.map_msg)
        # -----------------------------------------------------------------

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("수평이 정렬된 지도 노드가 시작되었습니다.")
        self.get_logger().info("aria_real_map.pgm 및 .yaml 파일이 성공적으로 생성되었습니다! Foxglove를 확인하세요.")

    def create_map_data(self):
        msg = OccupancyGrid()
        
        # 지도 해상도 및 크기 설정
        msg.info.resolution = 0.05
        width = 320
        height = 240
        msg.info.width = width
        msg.info.height = height

        # 맵의 원점을 중앙으로 맞춤
        msg.info.origin.position.x = -(width * msg.info.resolution) / 2.0
        msg.info.origin.position.y = -(height * msg.info.resolution) / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # 초기 맵 데이터 (0: 이동 가능)
        data = [0] * (width * height)

        # [민재님 원본 좌표 절대 유지]
        def draw_wall(x_min, x_max, y_min, y_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if 0 <= x < width and 0 <= y < height:
                        data[y * width + x] = 100

        # 1. 바깥쪽 U자 벽
        draw_wall(20, 260, 30, 35)    # 맨 아래 가로 벽
        draw_wall(20, 25, 30, 180)    # 왼쪽 바깥 세로 벽
        draw_wall(255, 260, 30, 180)  # 오른쪽 바깥 세로 벽

        # 2. 양쪽 세로 통로 위쪽 끝 마감 (지붕)
        draw_wall(20, 80, 175, 180)   # 왼쪽 방 지붕
        draw_wall(179, 260, 175, 180) # 오른쪽 방 지붕

        # 3. 안쪽 벽 (비대칭 공간 형성)
        draw_wall(80, 179, 100, 105)  # 중앙 가로 연결 벽
        draw_wall(75, 80, 75, 180)    # 왼쪽 안쪽 세로 벽 (왼쪽 방 17.5m² 확보)
        draw_wall(179, 184, 75, 180)  # 오른쪽 안쪽 세로 벽 (오른쪽 방 24.8m² 확보)

        # 4. 바깥쪽 문 (y=75~80 선상에 수평 고정)
        draw_wall(20, 35, 75, 80)     # 왼쪽 바깥 출입문
        draw_wall(245, 260, 75, 80)   # 오른쪽 바깥 출입문

        # 5. 안쪽 문 (바깥쪽 문과 동일한 y=75~80 선상에 수평 고정)
        draw_wall(60, 75, 75, 80)     # 왼쪽 안쪽 문
        draw_wall(184, 199, 75, 80)   # 오른쪽 안쪽 문

        msg.data = data
        return msg

    def save_map_to_files(self, msg):
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # --- [PGM 이미지 생성 로직] ---
        # ROS 맵 데이터(1D 배열)를 읽어 PGM 이미지 픽셀로 변환
        # 기본 배경은 205(회색, Unknown)로 설정
        img = Image.new('L', (width, height), 205)
        pixels = img.load()

        for y in range(height):
            for x in range(width):
                val = msg.data[y * width + x]
                # ROS 좌표계의 y=0은 맨 아래이므로, 이미지 저장 시 상하 반전 처리
                img_y = height - 1 - y 
                if val == 100:
                    pixels[x, img_y] = 0    # 벽 (검은색)
                elif val == 0:
                    pixels[x, img_y] = 254  # 이동 가능 공간 (흰색)

        # 파일 저장 (기존 파일명 유지)
        pgm_filename = 'aria_real_map.pgm'
        yaml_filename = 'aria_real_map.yaml'

        img.save(pgm_filename)

        # --- [YAML 파일 생성 로직] ---
        yaml_content = f"""image: {pgm_filename}
resolution: {res}
origin: [{origin_x}, {origin_y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(yaml_filename, 'w') as f:
            f.write(yaml_content)

    def timer_callback(self):
        # 1초마다 시간 스탬프만 갱신해서 계속 퍼블리시
        self.map_msg.header = Header()
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.header.frame_id = 'map'
        self.publisher_.publish(self.map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StaticMapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
