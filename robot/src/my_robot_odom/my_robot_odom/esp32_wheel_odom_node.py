import json
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import TransformStamped, Twist, Quaternion
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

# 펌웨어(main.cpp readTof0xDistance)와 맞춘 ToF 특수값
TOF_OUT_OF_RANGE = -2.0   # 범위 밖(비어 있음)
TOF_UNKNOWN = -1.0        # 미완료/타임아웃/고장


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


class Esp32WheelOdomNode(Node):
    def __init__(self):
        super().__init__("esp32_wheel_odom_node")

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tof_pub  = self.create_publisher(LaserScan, "/scan_tof", 10)
        self.imu_pub  = self.create_publisher(Imu, "/imu/data", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # ── ToF → LaserScan 변환 설정 ──
        # 펌웨어 기준 배치: ToF1=좌, ToF2=우, ToF3=중앙 (main.cpp XSHUT 핀 주석).
        # ROS 규약상 각도는 +가 왼쪽(반시계), -가 오른쪽이다.
        # 실제 장착 각도가 다르면 아래 파라미터만 바꾸면 된다.
        self.declare_parameter('tof1_angle_deg', 45.0)    # 좌
        self.declare_parameter('tof2_angle_deg', -45.0)   # 우
        self.declare_parameter('tof3_angle_deg', 0.0)     # 중앙
        self.declare_parameter('tof_fov_deg', 25.0)       # VL53L0X 시야각
        self.declare_parameter('tof_ray_step_deg', 2.5)   # 시야각을 몇 도 간격 빔으로 채울지
        self.declare_parameter('tof_range_max', 1.2)      # VL53L0X 기본 모드 신뢰 거리(m)
        self.tof_angles = [math.radians(self.get_parameter(f'tof{i}_angle_deg').value) for i in (1, 2, 3)]
        self.tof_half_fov = math.radians(self.get_parameter('tof_fov_deg').value) / 2.0
        self.tof_step = math.radians(self.get_parameter('tof_ray_step_deg').value)
        self.tof_range_max = float(self.get_parameter('tof_range_max').value)

        self.wheel_base   = 0.2475
        self.speed_to_pwm = 350.0
        self.left_coeff   = 0.97

        self.target_l_speed = 0
        self.target_r_speed = 0
        self.cmd_updated = False

        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.v, self.w = 0.0, 0.0

        # 브리지에서 오는 패킷 구독 (serial 직접 열지 않음)
        self.create_subscription(String, "/esp32/odom_raw", self.odom_raw_callback, 10)
        self.create_subscription(String, "/esp32/nav_raw",  self.nav_raw_callback,  10)

        # /cmd_vel → 모터 명령 변환 후 브리지로 전달
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_sub_callback, 10)
        self.motor_cmd_pub = self.create_publisher(String, "/esp32/motor_cmd", 10)

        self.timer = self.create_timer(0.05, self.timer_callback)

    # ──────────────────────────────────────────────
    # 브리지 토픽 콜백
    # ──────────────────────────────────────────────

    def odom_raw_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
            self.x     = float(packet.get("posX",     self.x))
            self.y     = float(packet.get("posY",     self.y))
            self.theta = float(packet.get("posTheta", self.theta))
            self.v     = float(packet["vel"]["lin"])
            self.w     = float(packet["vel"]["ang"])
        except Exception as e:
            self.get_logger().warn(f"odom_raw 파싱 실패: {e}")

    def nav_raw_callback(self, msg: String):
        try:
            packet = json.loads(msg.data)
            now = self.get_clock().now().to_msg()
            self.publish_tof_scan(packet, now)
            self.publish_imu(packet, now)
        except Exception as e:
            self.get_logger().warn(f"nav_raw 파싱 실패: {e}")

    # ──────────────────────────────────────────────
    # /cmd_vel → /esp32/motor_cmd
    # ──────────────────────────────────────────────

    def cmd_vel_sub_callback(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z *0.5

        v_left  = v - (w * self.wheel_base / 2.0)
        v_right = v + (w * self.wheel_base / 2.0)

        pwm_left  = v_left  * self.speed_to_pwm * self.left_coeff
        pwm_right = v_right * self.speed_to_pwm

        min_pwm = 40
        if   0.1  < pwm_left  < min_pwm:  pwm_left  = min_pwm
        elif -min_pwm < pwm_left  < -0.1: pwm_left  = -min_pwm
        if   0.1  < pwm_right < min_pwm:  pwm_right = min_pwm
        elif -min_pwm < pwm_right < -0.1: pwm_right = -min_pwm

        self.target_l_speed = int(max(min(round(pwm_left),  255), -255))
        self.target_r_speed = int(max(min(round(pwm_right), 255), -255))
        self.cmd_updated = True

    # ──────────────────────────────────────────────
    # 50ms 타이머: 모터 명령 발행 + TF/Odom 발행
    # ──────────────────────────────────────────────

    def timer_callback(self):
        if self.cmd_updated:
            cmd = {"left": self.target_l_speed, "right": self.target_r_speed, "mode": 1}
            motor_msg = String()
            motor_msg.data = json.dumps(cmd)
            self.motor_cmd_pub.publish(motor_msg)
            self.cmd_updated = False

        now = self.get_clock().now().to_msg()
        tfs = []

        # EKF 필터 충돌 방지를 위해 odom TF 발행 주석 처리
        # t_odom = TransformStamped()
        # t_odom.header.stamp = now
        # t_odom.header.frame_id = "odom"
        # t_odom.child_frame_id = "base_link"
        # t_odom.transform.translation.x = self.x
        # t_odom.transform.translation.y = self.y
        # t_odom.transform.rotation = yaw_to_quaternion(self.theta)
        # tfs.append(t_odom)

        t_laser = TransformStamped()
        t_laser.header.stamp      = now
        t_laser.header.frame_id   = "base_link"
        t_laser.child_frame_id    = "laser_frame"
        t_laser.transform.translation.x = -0.04
        t_laser.transform.translation.y =  0.0
        t_laser.transform.translation.z =  0.440
        t_laser.transform.rotation.w    =  1.0
        tfs.append(t_laser)

        t_tof = TransformStamped()
        t_tof.header.stamp      = now
        t_tof.header.frame_id   = "base_link"
        t_tof.child_frame_id    = "tof_frame"
        t_tof.transform.translation.x = 0.13
        t_tof.transform.translation.y = 0.0
        t_tof.transform.translation.z = 0.01
        t_tof.transform.rotation.w    = 1.0
        tfs.append(t_tof)

        self.tf_broadcaster.sendTransform(tfs)

        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.orientation = yaw_to_quaternion(self.theta)
        odom.twist.twist.linear.x  = self.v
        odom.twist.twist.angular.z = self.w

        odom.pose.covariance[0]  = 0.01
        odom.pose.covariance[7]  = 0.01
        odom.pose.covariance[35] = 0.05

        self.odom_pub.publish(odom)

    # ──────────────────────────────────────────────
    # ToF / IMU 발행
    # ──────────────────────────────────────────────

    def publish_tof_scan(self, packet, now):
        """ToF 3개를 Nav2가 읽을 수 있는 LaserScan(/scan_tof)으로 변환한다.

        이전 버전의 문제
          - 센서 1개당 빔 1개(점 3개)뿐이라, 25° 시야각 안의 장애물을 점으로만 표시했다.
          - ranges 순서가 [ToF1(좌), ToF3, ToF2(우)]인데 angle_min이 -45°(오른쪽)라
            좌우가 뒤바뀌어 있었다.
        현재
          - 각 센서의 시야각(±12.5°)을 tof_ray_step_deg 간격 빔으로 채운다.
          - 센서 사이 빈 구간은 NaN → Nav2가 무시(표시도 지우지도 않음).
          - 펌웨어 값의 의미를 구분한다(2026-10 수정).
              > 0   : 측정 거리(mm)
              -2    : 범위 밖(앞이 비어 있음) → inf → 장애물 표시를 지운다(inf_is_valid)
              -1 등 : 측정 미완료·타임아웃·센서 고장 → NaN → 표시도 지우지도 않는다
            이전엔 -1을 모두 inf로 바꿔서, 센서가 고장 나거나 값이 늦게 오면
            '앞이 비어 있다'고 판단해 실제 장애물 표시까지 지워버렸다.
            (펌웨어를 새로 굽기 전에는 -2가 오지 않으므로 '지우기'는 라이다가 담당한다.)
        """
        readings = []
        for key in ('tof1', 'tof2', 'tof3'):
            raw = float(packet.get(key, -1.0))
            if raw > 0:
                readings.append(raw / 1000.0)   # mm → m
            elif raw == TOF_OUT_OF_RANGE:
                readings.append(float('inf'))   # 확실히 비어 있음
            else:
                readings.append(float('nan'))   # 모름 — 판단 보류

        lo = min(self.tof_angles) - self.tof_half_fov
        hi = max(self.tof_angles) + self.tof_half_fov
        n = int(round((hi - lo) / self.tof_step)) + 1

        ranges = []
        for i in range(n):
            ang = lo + i * self.tof_step
            value = float('nan')
            for center, dist in zip(self.tof_angles, readings):
                if abs(ang - center) <= self.tof_half_fov + 1e-6:
                    # 시야각이 겹치면 더 가까운 값을 쓴다(장애물 우선).
                    value = dist if math.isnan(value) else min(value, dist)
            ranges.append(value)

        msg = LaserScan()
        msg.header.stamp    = now
        msg.header.frame_id = "tof_frame"
        msg.angle_min       = lo
        msg.angle_max       = lo + (n - 1) * self.tof_step
        msg.angle_increment = self.tof_step
        msg.scan_time       = 0.1          # NavPacket 10Hz
        msg.range_min       = 0.02
        msg.range_max       = self.tof_range_max
        msg.ranges = ranges
        self.tof_pub.publish(msg)

    def publish_imu(self, packet, now):
        imu_msg = Imu()
        imu_msg.header.stamp    = now
        imu_msg.header.frame_id = "base_link"

        gyro_data = packet.get("gyro", {})
        imu_msg.angular_velocity.x = float(gyro_data.get("x", 0.0))
        imu_msg.angular_velocity.y = float(gyro_data.get("y", 0.0))
        imu_msg.angular_velocity.z = float(gyro_data.get("z", 0.0))

        self.imu_pub.publish(imu_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Esp32WheelOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
