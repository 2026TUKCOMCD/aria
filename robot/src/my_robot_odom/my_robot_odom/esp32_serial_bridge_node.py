import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .packet_parser import PacketParser


class Esp32SerialBridgeNode(Node):
    """/dev/serial0 을 독점해서 열고, 파싱된 패킷을 ROS2 토픽으로 배포.
    모터 명령은 /esp32/motor_cmd 토픽을 구독해서 ESP32로 전달.

    발행 토픽:
        /esp32/odom_raw  (std_msgs/String, JSON)
        /esp32/air_raw   (std_msgs/String, JSON)
        /esp32/nav_raw   (std_msgs/String, JSON)

    구독 토픽:
        /esp32/motor_cmd (std_msgs/String, JSON: {"left": int, "right": int, "mode": int})
    """

    def __init__(self):
        super().__init__("esp32_serial_bridge_node")

        self.declare_parameter("port", "/dev/serial0")
        self.declare_parameter("baudrate", 115200)

        port     = self.get_parameter("port").value
        baudrate = int(self.get_parameter("baudrate").value)

        self.parser = PacketParser(port=port, baudrate=baudrate)

        # ESP32 → ROS2
        self.odom_pub = self.create_publisher(String, "/esp32/odom_raw", 10)
        self.air_pub  = self.create_publisher(String, "/esp32/air_raw",  10)
        self.nav_pub  = self.create_publisher(String, "/esp32/nav_raw",  10)

        # ROS2 → ESP32
        self.create_subscription(String, "/esp32/motor_cmd", self.motor_cmd_callback, 10)
        self.create_subscription(String, "/esp32/fan_cmd", self.fan_cmd_callback, 10)

        # 20ms 주기로 시리얼 버퍼 읽기
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info(f"Serial bridge started: {port} @ {baudrate}")

    def timer_callback(self):
        packet = self.parser.read_packet()
        if packet is None:
            return

        p_type = packet.get("type")
        packet.pop("timestamp", None)  # datetime은 JSON 직렬화 불가

        msg = String()
        msg.data = json.dumps(packet)

        if p_type == "ODOM":
            self.odom_pub.publish(msg)
        elif p_type == "AIR":
            self.air_pub.publish(msg)
        elif p_type == "NAV":
            self.nav_pub.publish(msg)

    def motor_cmd_callback(self, msg: String):
        try:
            cmd   = json.loads(msg.data)
            left  = int(cmd.get("left",  0))
            right = int(cmd.get("right", 0))
            mode  = int(cmd.get("mode",  self.parser.MODE_MANUAL))
            self.parser.send_motor_command(left, right, mode=mode)
        except Exception as e:
            self.get_logger().error(f"모터 명령 처리 실패: {e}")

    def fan_cmd_callback(self, msg: String):
        try:
            cmd = json.loads(msg.data)
            # JSON 데이터에서 "fan" 값을 꺼내옵니다. (없으면 0)
            fan = int(cmd.get("fan", 0))
            # 파서의 팬 전송 함수를 호출합니다.
            self.parser.send_fan_command(fan_speed=fan)
            self.get_logger().info(f"브리지 노드: 팬 명령 전달 완료 ({fan})")
        except Exception as e:
            self.get_logger().error(f"팬 명령 처리 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Esp32SerialBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.parser.send_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
