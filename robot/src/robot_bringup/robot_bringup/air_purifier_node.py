import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class AirPurifierNode(Node):
    def __init__(self):
        super().__init__('air_purifier_node')

        # [구독] 센서 데이터 (ESP32 -> ROS 2)
        self.air_sub = self.create_subscription(
            String,
            '/esp32/air_raw',
            self.air_callback,
            10
        )

        # [발행] 모터와 팬 제어를 각각 독립된 토픽으로 분리
        self.motor_pub = self.create_publisher(String, '/esp32/motor_cmd', 10)
        self.fan_pub = self.create_publisher(String, '/esp32/fan_cmd', 10)

        # 상태 관리 변수 및 판단 기준치 (PM2.5)
        self.is_purifying = False
        self.bad_threshold = 35.0  # 집중 정화(로봇 정지)를 시작할 기준치
        self.good_threshold = 15.0 # 주행을 재개할 깨끗한 기준치

        self.get_logger().info("Air Purifier Node 시작 완료: 팬과 모터 통신이 완벽히 분리되었습니다.")

    def air_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            pm25 = data.get('pm25', -1.0)

            # 센서 초기화 전(-1.0)이거나 데이터가 비정상인 경우 스킵
            if pm25 < 0:
                return

            # 1. 미세먼지 수치가 기준치 이상일 때 (공기질 악화)
            if pm25 >= self.bad_threshold and not self.is_purifying:
                self.get_logger().warn(f"🚨 공기 오염 감지 (PM2.5: {pm25})! 주행을 멈추고 팬을 최대로 가동합니다.")
                self.is_purifying = True

                # 모터 정지 명령 (mode 1: 수동 제어 모드에서 0 속도)
                self.send_motor_cmd(left=0, right=0, mode=1)

                # 팬 최대 가동 명령 (255)
                self.send_fan_cmd(fan=255)

            # 2. 미세먼지 수치가 기준치 미만으로 떨어졌을 때 (공기질 회복)
            elif pm25 < self.good_threshold and self.is_purifying:
                self.get_logger().info(f"✅ 공기질 회복 완료 (PM2.5: {pm25}). 팬 전원을 완전히 차단합니다.")
                self.is_purifying = False

                # ★ 핵심: 평시 팬 출력을 77이 아닌 0으로 변경하여 완전히 끔
                self.send_fan_cmd(fan=0)

        except Exception as e:
            self.get_logger().error(f"공기질 데이터 파싱 실패: {e}")

    def send_motor_cmd(self, left: int, right: int, mode: int):
        """바퀴(모터) 전용 제어 명령 전송"""
        msg = String()
        msg.data = json.dumps({"left": left, "right": right, "mode": mode})
        self.motor_pub.publish(msg)

    def send_fan_cmd(self, fan: int):
        """팬 전용 제어 명령 전송"""
        msg = String()
        msg.data = json.dumps({"fan": fan})
        self.fan_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AirPurifierNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 노드를 강제 종료할 때 로봇과 팬을 안전하게 멈추는 페일세이프 로직
        node.get_logger().info("노드 종료 중... 팬과 모터를 정지합니다.")
        node.send_motor_cmd(left=0, right=0, mode=0) # 비상 정지 모드(0)
        node.send_fan_cmd(fan=0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
