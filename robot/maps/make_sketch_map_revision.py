#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from PIL import Image  

class StaticMapNode(Node):
    def __init__(self):
        super().__init__('aria_static_map_node')
        
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)
        
        # --- 맵 데이터 생성 및 파일 저장 (노드 시작 시 1회만 실행) ---
        self.map_msg = self.create_map_data()
        self.save_map_to_files(self.map_msg)
        # -----------------------------------------------------------------

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("17.5m^2 및 24.8m^2 면적이 정확히 반영된 다이아몬드 지도가 시작되었습니다.")
        self.get_logger().info("aria_real_map.pgm 및 .yaml 파일이 성공적으로 생성되었습니다!")

    def create_map_data(self):
        msg = OccupancyGrid()
        
        # ==========================================
        # ★ 해상도 최적화 및 캔버스 크기 조정 ★
        # ==========================================
        msg.info.resolution = 0.05 # 1픽셀 = 5cm

        width = 400  # 20미터 폭
        height = 300 # 15미터 폭

        msg.info.width = width
        msg.info.height = height

        # 맵의 원점을 중앙으로 맞춤
        msg.info.origin.position.x = -(width * msg.info.resolution) / 2.0
        msg.info.origin.position.y = -(height * msg.info.resolution) / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # 초기 맵 데이터 (0: 이동 가능)
        data = [0] * (width * height)

        def set_cell(cx, cy, val):
            if 0 <= cx < width and 0 <= cy < height:
                data[cy * width + cx] = val

        # 뼈대 선이 없는 완벽한 기울어진 정사각형(마름모) 도장
        def draw_black_diamond(cx, cy):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) <= 2:
                        set_cell(cx + dx, cy + dy, 100)

        # 다이아몬드 도장을 선을 따라 찍는 함수
        def draw_wall(x_start, x_end, y_start, y_end):
            is_horizontal = (x_end - x_start) >= (y_end - y_start)
            step = 4  # 마름모 모서리가 빈틈없이 딱 맞물리는 간격
            
            if is_horizontal:
                for x in range(x_start, x_end + 1):
                    if (x - x_start) % step == 0:
                        draw_black_diamond(x, y_start)
            else:
                for y in range(y_start, y_end + 1):
                    if (y - y_start) % step == 0:
                        draw_black_diamond(x_start, y)

        # ========================================================
        # [정확한 면적 및 도면 비율이 반영된 맵 그리기] 
        # ========================================================
        
        # 1. 외곽 프레임
        draw_wall(20, 360, 30, 30)   # 하단 복도 바깥벽
        draw_wall(20, 20, 30, 230)   # 왼쪽 전체 바깥벽
        draw_wall(360, 360, 30, 230) # 오른쪽 전체 바깥벽
        draw_wall(20, 360, 230, 230) # 상단 전체 지붕

        # 2. 내측 세로벽 (방을 구분하는 벽)
        # 깊이(Y)는 모두 90 ~ 230 (140px = 7.0m)로 동일합니다.
        draw_wall(70, 70, 90, 230)   # 17.5㎡ 방 우측 벽 (폭 50px = 2.5m)
        draw_wall(289, 289, 90, 230) # 24.8㎡ 방 좌측 벽 (폭 71px = 3.55m)

        # 중앙부의 방 3개 (416, 418, 420호)를 나누는 세로벽
        draw_wall(143, 143, 90, 230)
        draw_wall(216, 216, 90, 230)

        # 3. 복도 가로벽 및 문(입구) 뚫기
        # 문이 들어갈 자리를 비워두고 양옆 벽만 그립니다 (문 폭: 16px = 0.8m)
        
        # 좌측 방 (17.5㎡) 입구
        draw_wall(20, 40, 90, 90)
        draw_wall(56, 70, 90, 90)

        # 중앙 방 1 (416호) 입구
        draw_wall(70, 100, 90, 90)
        draw_wall(116, 143, 90, 90)

        # 중앙 방 2 (418호) 입구
        draw_wall(143, 173, 90, 90)
        draw_wall(189, 216, 90, 90)

        # 중앙 방 3 (420호) 입구
        draw_wall(216, 246, 90, 90)
        draw_wall(262, 289, 90, 90)

        # 우측 방 (24.8㎡) 입구
        draw_wall(289, 310, 90, 90)
        draw_wall(326, 360, 90, 90)

        msg.data = data
        return msg

    def save_map_to_files(self, msg):
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # ROS 맵 데이터(1D 배열)를 읽어 PGM 이미지 픽셀로 변환
        img = Image.new('L', (width, height), 205)
        pixels = img.load()

        for y in range(height):
            for x in range(width):
                val = msg.data[y * width + x]
                img_y = height - 1 - y 
                if val == 100:
                    pixels[x, img_y] = 0    # 벽 (검은색)
                elif val == 0:
                    pixels[x, img_y] = 254  # 이동 가능 공간 (흰색)

        # 파일 저장
        pgm_filename = 'aria_real_map.pgm'
        yaml_filename = 'aria_real_map.yaml'

        img.save(pgm_filename)

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
