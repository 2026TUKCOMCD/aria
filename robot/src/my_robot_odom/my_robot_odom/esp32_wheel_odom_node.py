import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

from .packet_parser import PacketParser


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


class Esp32WheelOdomNode(Node):
    def __init__(self):
        super().__init__("esp32_wheel_odom_node")

        self.declare_parameter("port", "/dev/serial0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        port = self.get_parameter("port").value
        baudrate = int(self.get_parameter("baudrate").value)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.parser = PacketParser(port=port, baudrate=baudrate)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info(
            f"ESP32 odom node started on {port} @ {baudrate}"
        )

    def timer_callback(self):
        packet = self.parser.read_packet()
        self.get_logger().info(f"packet={packet}")

        if packet is None:
            return

        if packet.get("type") != "ODOM":
            return

        now_msg = self.get_clock().now().to_msg()

        x = float(packet["pos"]["x"])
        y = float(packet["pos"]["y"])
        theta = float(packet["pos"]["theta"])
        v = float(packet["vel"]["lin"])
        w = float(packet["vel"]["ang"])

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quaternion(theta)

        odom.twist.twist.linear.x = v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = w

        self.get_logger().info(
            f"publishing odom x={x:.3f}, y={y:.3f}, theta={theta:.3f}, v={v:.3f}, w={w:.3f}"
        )
        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now_msg
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame
            tf_msg.transform.translation.x = x
            tf_msg.transform.translation.y = y
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = yaw_to_quaternion(theta)
            self.tf_broadcaster.sendTransform(tf_msg)


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
