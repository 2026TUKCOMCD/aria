#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan  # ⭐️ ToF 센서 피드를 위한 레이저스캔 메시지 추가
import serial
import struct
import math

class ESP32BridgeNode(Node):
    def __init__(self):
        super().__init__('esp32_bridge_node')
        
        self.WHEEL_BASE = 0.23
        self.MAX_VEL_X = 0.35 
        
        # 1. 시리얼 포트 연결
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01)
            self.get_logger().info("✓ ESP32-S3 Serial Port Connected!")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to open serial port: {e}")
            
        # 2. 퍼블리셔 & 서브스크라이버 설정을 수신부 스펙과 매핑
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_tof', 10) # ⭐️ Nav2 obstacle_layer가 바라보는 토픽
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # 20ms 주기(50Hz)로 시리얼 포트 역직렬화 감시
        self.create_timer(0.02, self.read_serial)

    def read_serial(self):
        if not hasattr(self, 'ser') or not self.ser.is_open:
            return
            
        # 가변 패킷 파싱을 위해 헤더(2B)와 ID(1B)가 들어올 때까지 대기
        if self.ser.in_waiting >= 3:
            header = self.ser.read(2)
            if header == b'\x55\xAA': # 리틀엔디안 정렬 기준 0xAA55
                packet_id = self.ser.read(1)[0]
                
                # ① OdomPacket 처리 (전체 34바이트 중 3바이트 읽었으므로 남은 31바이트 파싱)
                if packet_id == 3 and self.ser.in_waiting >= 31:
                    payload = self.ser.read(31)
                    self.parse_odom_packet(payload)
                    
                # ② NavPacket 처리 (전체 42바이트 중 3바이트 읽었으므로 남은 39바이트 파싱)
                elif packet_id == 0 and self.ser.in_waiting >= 39:
                    payload = self.ser.read(39)
                    self.parse_nav_packet(payload)

    def parse_odom_packet(self, payload):
        unpacked = struct.unpack("<iifffffBH", payload)
        left_enc, right_enc, px, py, ptheta, lin_vel, ang_vel, checksum, tail = unpacked
        
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        
        q = self.euler_to_quaternion(0, 0, ptheta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        odom.twist.twist.linear.x = lin_vel
        odom.twist.twist.angular.z = ang_vel
        self.odom_pub.publish(odom)

    def parse_nav_packet(self, payload):
        # float 9개(ToF 3개, IMU 6개), checksum(B), tail(H) = 39바이트 언팩
        unpacked = struct.unpack("<fffffffffBH", payload)
        tof1, tof2, tof3 = unpacked[0], unpacked[1], unpacked[2]
        
        # [후처리 파이프라인 ①, ②]: mm 단위를 m 단위로 변환 및 예외 처리(Clamping)
        # 센서 에러 값(-1.0)이 들어오면 신뢰 거리를 넘는 무한대(inf) 처리하여 유령 벽 제거
        tof1_m = tof1 / 1000.0 if tof1 > 0 else float('inf') # 전방
        tof2_m = tof2 / 1000.0 if tof2 > 0 else float('inf') # 좌측
        tof3_m = tof3 / 1000.0 if tof3 > 0 else float('inf') # 우측
        
        # ROS 2 표준형 실시간 레이저 스캔 데이터 생성
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_link' # 로봇 중심 좌표계 기준
        
        # 물리 센서 3개의 각도를 시분할 배열 매핑
        # Index 0: 우측(-90도), Index 1: 전방(0도), Index 2: 좌측(90도)
        scan.angle_min = -1.5708  # -90도 (rad)
        scan.angle_max = 1.5708   # 90도 (rad)
        scan.angle_increment = 1.5708 # 90도 간격 격자
        scan.range_min = 0.1
        scan.range_max = 1.2       # YAML 파일의 obstacle_max_range와 매칭
        
        scan.ranges = [tof3_m, tof1_m, tof2_m] # [우, 전, 좌] 순서대로 주입
        
        self.scan_pub.publish(scan)

    def cmd_vel_callback(self, msg):
        lin_x = msg.linear.x
        ang_z = msg.angular.z
        
        v_left = lin_x - (ang_z * self.WHEEL_BASE / 2.0)
        v_right = lin_x + (ang_z * self.WHEEL_BASE / 2.0)
        
        left_speed = int((v_left / self.MAX_VEL_X) * 255)
        right_speed = int((v_right / self.MAX_VEL_X) * 255)
        
        left_speed = max(min(left_speed, 255), -255)
        right_speed = max(min(right_speed, 255), -255)
        
        header, packet_id, mode, tail = 0xAA55, 2, 2, 0x0A0D
        temp_buf = struct.pack("<HBBhh", header, packet_id, mode, left_speed, right_speed)
        checksum = sum(temp_buf) & 0xFF
        
        packet = struct.pack("<HBBhhBH", header, packet_id, mode, left_speed, right_speed, checksum, tail)
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.write(packet)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ]

def main(args=None):
    rclpy.init(args=args)
    node = ESP32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
