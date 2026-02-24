#!/usr/bin/env python3
import math
import struct
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

import serial


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


class Esp32WheelOdomNode(Node):
    """
    ESP32 OdomPacket(id=3) 수신 -> left/right encoder count 기반으로 wheel odom 계산 -> /odom + TF 발행

    ESP32 packed struct (little-endian):
      uint16 header = 0xAA55   (wire bytes: 55 AA)
      uint8  id = 3
      int32  leftEncoderCount
      int32  rightEncoderCount
      float  posX
      float  posY
      float  posTheta
      float  linearVel
      float  angularVel
      uint8  checksum
      uint16 tail = 0x0A0D     (wire bytes: 0D 0A)
    Total size = 34 bytes
    """

    HEADER_BYTES = b"\xAA\x55"  # little-endian 0xAA55 on wire
    TAIL_BYTES = b"\x0A\x0D"    # little-endian 0x0A0D on wire
    PACKET_SIZE = 34
    ODOM_ID = 3

    # struct format (little-endian):
    # H B i i f f f f f B H
    STRUCT_FMT = "<HBiifffffBH"

    def __init__(self):
        super().__init__("esp32_wheel_odom")

        # ===== Parameters =====
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("wheel_diameter", 0.065)     # meters
        self.declare_parameter("wheel_base", 0.15)          # meters
        self.declare_parameter("counts_per_rev", 360.0)     # TODO: 실측값으로 바꾸기
        self.declare_parameter("left_sign", 1)
        self.declare_parameter("right_sign", 1)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("min_dt", 1e-4)
        self.declare_parameter("count_deadband", 0)         # 정지 노이즈 필터 (예: 0~2)
        self.declare_parameter("serial_timeout", 0.05)

        self.port = self.get_parameter("port").value
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.wheel_diameter = float(self.get_parameter("wheel_diameter").value)
        self.wheel_base = float(self.get_parameter("wheel_base").value)
        self.counts_per_rev = float(self.get_parameter("counts_per_rev").value)
        self.left_sign = int(self.get_parameter("left_sign").value)
        self.right_sign = int(self.get_parameter("right_sign").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.min_dt = float(self.get_parameter("min_dt").value)
        self.count_deadband = int(self.get_parameter("count_deadband").value)
        self.serial_timeout = float(self.get_parameter("serial_timeout").value)

        # ===== ROS publishers =====
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===== State =====
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.prev_left_count: Optional[int] = None
        self.prev_right_count: Optional[int] = None
        self.prev_time: Optional[float] = None

        self.last_esp32_pose = None  # (x, y, theta)
        self.last_esp32_vel = None   # (v, w)

        self._ser = None
        self._rx_thread = None
        self._running = True
        self._rx_buffer = bytearray()

        self._open_serial()
        self._start_rx_thread()

        self.get_logger().info(
            f"Started ESP32 wheel odom node | port={self.port}, baud={self.baudrate}, "
            f"wheel_d={self.wheel_diameter}, wheel_base={self.wheel_base}, cpr={self.counts_per_rev}, "
            f"signs=({self.left_sign},{self.right_sign})"
        )

    # ---------------- Serial ----------------

    def _open_serial(self):
        try:
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.serial_timeout,
            )
            self.get_logger().info(f"Opened serial: {self.port} @ {self.baudrate}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.port}: {e}")
            raise

    def _start_rx_thread(self):
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _rx_loop(self):
        while self._running and rclpy.ok():
            try:
                data = self._ser.read(256)
                if data:
                    self._rx_buffer.extend(data)
                    self._parse_buffer()
            except Exception as e:
                self.get_logger().error(f"Serial read error: {e}")
                time.sleep(0.2)

    def _parse_buffer(self):
        while True:
            # Find header
            idx = self._rx_buffer.find(self.HEADER_BYTES)
            if idx < 0:
                # keep only last 1 byte (in case header splits)
                if len(self._rx_buffer) > 1:
                    self._rx_buffer = self._rx_buffer[-1:]
                return

            # discard preceding junk
            if idx > 0:
                del self._rx_buffer[:idx]

            if len(self._rx_buffer) < self.PACKET_SIZE:
                return

            pkt = bytes(self._rx_buffer[:self.PACKET_SIZE])

            # Quick tail check
            if pkt[-2:] != self.TAIL_BYTES:
                # bad alignment, drop first byte and retry
                del self._rx_buffer[0]
                continue

            # Full unpack
            try:
                unpacked = struct.unpack(self.STRUCT_FMT, pkt)
            except struct.error:
                del self._rx_buffer[0]
                continue

            header, pkt_id, left_cnt, right_cnt, esp_x, esp_y, esp_th, esp_v, esp_w, checksum, tail = unpacked

            # Validate header/tail values (endianness-safe after unpack)
            if header != 0xAA55 or tail != 0x0A0D:
                del self._rx_buffer[0]
                continue

            # Checksum validation (ESP32 sums first len-3 bytes)
            expected_checksum = self._calculate_checksum(pkt)
            if checksum != expected_checksum:
                self.get_logger().warn(
                    f"[DEBUG] Checksum mismatch (ignored): got={checksum}, expected={expected_checksum}"
                )
                #del self._rx_buffer[:self.PACKET_SIZE]
                #continue

            # Consume packet
            del self._rx_buffer[:self.PACKET_SIZE]

            # Only process OdomPacket
            if pkt_id == self.ODOM_ID:
                self.get_logger().info(f"[DEBUG] ODOM packet parsed: L={left_cnt}, R={right_cnt}")

            self.last_esp32_pose = (esp_x, esp_y, esp_th)
            self.last_esp32_vel = (esp_v, esp_w)

            self._handle_encoder_counts(left_cnt, right_cnt)

    @staticmethod
    def _calculate_checksum(pkt: bytes) -> int:
        # ESP32 calculateChecksum(data, len): sums bytes [0 .. len-4] because len-3 excludes checksum+tail(2)
        # for len=34 => sum first 31 bytes, i.e., pkt[:-3]
        return sum(pkt[:-3]) & 0xFF

    # ---------------- Odom ----------------

    def _handle_encoder_counts(self, raw_left_count: int, raw_right_count: int):
        now = time.time()

        # Apply sign correction
        left_count = self.left_sign * raw_left_count
        right_count = self.right_sign * raw_right_count

        if self.prev_left_count is None:
            self.prev_left_count = left_count
            self.prev_right_count = right_count
            self.prev_time = now
            return

        dt = now - self.prev_time
        if dt < self.min_dt:
            return

        dcl = left_count - self.prev_left_count
        dcr = right_count - self.prev_right_count

        # Optional deadband for encoder noise
        if abs(dcl) <= self.count_deadband:
            dcl = 0
        if abs(dcr) <= self.count_deadband:
            dcr = 0

        self.prev_left_count = left_count
        self.prev_right_count = right_count
        self.prev_time = now

        wheel_radius = self.wheel_diameter / 2.0
        dist_per_count = (2.0 * math.pi * wheel_radius) / self.counts_per_rev

        dL = dcl * dist_per_count
        dR = dcr * dist_per_count

        dS = (dR + dL) / 2.0
        dTheta = (dR - dL) / self.wheel_base

        # Midpoint integration
        self.x += dS * math.cos(self.theta + dTheta / 2.0)
        self.y += dS * math.sin(self.theta + dTheta / 2.0)
        self.theta += dTheta

        # Normalize
        while self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        while self.theta < -math.pi:
            self.theta += 2.0 * math.pi

        v = dS / dt
        w = dTheta / dt

        self._publish_odom(v, w)

    def _publish_odom(self, v: float, w: float):
        now_msg = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quaternion(self.theta)

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        # 초기값(대충) - 나중에 튜닝 가능
        odom.pose.covariance = [
            0.02, 0,    0,    0,    0,    0,
            0,    0.02, 0,    0,    0,    0,
            0,    0,    9999, 0,    0,    0,
            0,    0,    0,    9999, 0,    0,
            0,    0,    0,    0,    9999, 0,
            0,    0,    0,    0,    0,    0.05
        ]
        odom.twist.covariance = [
            0.05, 0,    0,    0,    0,    0,
            0,    0.05, 0,    0,    0,    0,
            0,    0,    9999, 0,    0,    0,
            0,    0,    0,    9999, 0,    0,
            0,    0,    0,    0,    9999, 0,
            0,    0,    0,    0,    0,    0.1
        ]

        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now_msg
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = float(self.x)
            tf.transform.translation.y = float(self.y)
            tf.transform.translation.z = 0.0
            tf.transform.rotation = yaw_to_quaternion(self.theta)
            self.tf_broadcaster.sendTransform(tf)

    def destroy_node(self):
        self._running = False
        try:
            if self._rx_thread is not None:
                self._rx_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._ser is not None and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


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


if __name__ == "__main__":
    main()
