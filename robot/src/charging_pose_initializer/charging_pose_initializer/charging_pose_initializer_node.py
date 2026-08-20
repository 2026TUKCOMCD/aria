#!/usr/bin/env python3

import json
import math
import os
import statistics
import urllib.error
import urllib.request

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


class ChargingPoseInitializer(Node):
    """
    충전 위치 저장 및 AMCL 자동 초기화 노드.

    기능 1:
      /save_dock_pose 서비스 호출
      -> map -> base_link TF를 여러 번 읽음
      -> x, y, theta 평균 계산
      -> POST /robots/1/dock 로 충전 위치 저장

    기능 2:
      auto_initialize=true
      -> GET /robots/1/dock 로 충전 위치 조회
      -> /initialpose 발행
      -> AMCL 초기 위치 자동 설정
    """

    def __init__(self):
        super().__init__('charging_pose_initializer')

        # API Gateway 기본 주소
        self.declare_parameter(
            'base_api_url',
            'https://ph7ckbtbl3.execute-api.ap-northeast-2.amazonaws.com',
        )

        self.declare_parameter('robot_id', 1)

        # 토큰은 환경변수 ARIA_AUTH_TOKEN에서 가져옴
        self.declare_parameter(
            'authorization_token',
            os.environ.get('ARIA_AUTH_TOKEN', ''),
        )

        # 매핑 모드에서는 false, 자율주행 모드에서는 true
        self.declare_parameter('auto_initialize', True)

        # 충전 위치 저장 시 평균낼 TF 개수
        self.declare_parameter('sample_count', 10)

        # TF 샘플 간격
        self.declare_parameter('sample_interval', 0.2)

        # /initialpose 발행 횟수
        self.declare_parameter('initialpose_publish_count', 3)

        # 자동 초기화 재시도 간격
        self.declare_parameter('retry_interval', 2.0)

        self.base_api_url = (
            self.get_parameter('base_api_url')
            .get_parameter_value()
            .string_value
            .rstrip('/')
        )

        self.robot_id = (
            self.get_parameter('robot_id')
            .get_parameter_value()
            .integer_value
        )

        self.authorization_token = (
            self.get_parameter('authorization_token')
            .get_parameter_value()
            .string_value
        )

        self.auto_initialize = (
            self.get_parameter('auto_initialize')
            .get_parameter_value()
            .bool_value
        )

        self.sample_count = (
            self.get_parameter('sample_count')
            .get_parameter_value()
            .integer_value
        )

        self.sample_interval = (
            self.get_parameter('sample_interval')
            .get_parameter_value()
            .double_value
        )

        self.initialpose_publish_count = (
            self.get_parameter('initialpose_publish_count')
            .get_parameter_value()
            .integer_value
        )

        self.retry_interval = (
            self.get_parameter('retry_interval')
            .get_parameter_value()
            .double_value
        )

        self.api_url = (
            f'{self.base_api_url}/robots/{self.robot_id}/dock'
        )

        # TF 조회용
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # AMCL 초기 위치 발행
        self.initialpose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10,
        )

        # 충전 위치 저장 서비스
        self.save_dock_service = self.create_service(
            Trigger,
            '/save_dock_pose',
            self.save_dock_pose_callback,
        )

        self.charging_pose = None
        self.initialpose_publish_count_current = 0
        self.initialization_completed = False

        self.initialize_timer = None

        if self.auto_initialize:
            self.initialize_timer = self.create_timer(
                self.retry_interval,
                self.try_auto_initialize,
            )

        self.get_logger().info(
            'Charging pose initializer started'
        )

        self.get_logger().info(
            f'API URL: {self.api_url}'
        )

        self.get_logger().info(
            f'auto_initialize: {self.auto_initialize}'
        )

    def get_request_headers(self):
        if not self.authorization_token:
            raise RuntimeError(
                'ARIA_AUTH_TOKEN 환경변수가 비어 있습니다.'
            )

        return {
            'Authorization': self.authorization_token,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        """
        Quaternion을 yaw 라디안 값으로 변환.
        """

        siny_cosp = 2.0 * (
            w * z + x * y
        )

        cosy_cosp = 1.0 - 2.0 * (
            y * y + z * z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp,
        )

    def lookup_current_map_pose(self):
        """
        map -> base_link TF를 읽어서
        지도 기준 현재 로봇 위치와 방향을 반환.
        """

        transform = self.tf_buffer.lookup_transform(
            'map',
            'base_link',
            rclpy.time.Time(),
            timeout=Duration(seconds=2.0),
        )

        translation = (
            transform.transform.translation
        )

        rotation = (
            transform.transform.rotation
        )

        yaw = self.quaternion_to_yaw(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        return {
            'x': float(translation.x),
            'y': float(translation.y),
            'theta': float(yaw),
        }

    def calculate_average_pose(self):
        """
        map -> base_link TF를 여러 번 읽어서 평균 계산.

        yaw는 일반 평균이 아니라 sin/cos 평균을 사용.
        예를 들어 179도와 -179도를 평균내도 0도가 되지 않도록 함.
        """

        x_values = []
        y_values = []
        yaw_sin_values = []
        yaw_cos_values = []

        for index in range(self.sample_count):
            pose = self.lookup_current_map_pose()

            x_values.append(
                pose['x']
            )

            y_values.append(
                pose['y']
            )

            yaw_sin_values.append(
                math.sin(pose['theta'])
            )

            yaw_cos_values.append(
                math.cos(pose['theta'])
            )

            self.get_logger().info(
                f'TF sample {index + 1}/{self.sample_count}: '
                f"x={pose['x']:.3f}, "
                f"y={pose['y']:.3f}, "
                f"theta={pose['theta']:.3f}"
            )

            # ROS callback 처리 및 샘플 간격
            rclpy.spin_once(
                self,
                timeout_sec=self.sample_interval,
            )

        average_x = statistics.fmean(
            x_values
        )

        average_y = statistics.fmean(
            y_values
        )

        average_theta = math.atan2(
            statistics.fmean(yaw_sin_values),
            statistics.fmean(yaw_cos_values),
        )

        return {
            'x': average_x,
            'y': average_y,
            'theta': average_theta,
        }

    def post_dock_pose(self, pose):
        """
        현재 충전 위치를 Lambda API에 POST 저장.
        """

        request_body = json.dumps({
            'x': pose['x'],
            'y': pose['y'],
            'theta': pose['theta'],
        }).encode('utf-8')

        request = urllib.request.Request(
            self.api_url,
            data=request_body,
            method='POST',
            headers=self.get_request_headers(),
        )

        with urllib.request.urlopen(
            request,
            timeout=8,
        ) as response:
            response_body = (
                response.read().decode('utf-8')
            )

            return json.loads(
                response_body
            )

    def get_dock_pose(self):
        """
        Lambda API에서 저장된 충전 위치 조회.
        """

        request = urllib.request.Request(
            self.api_url,
            method='GET',
            headers=self.get_request_headers(),
        )

        with urllib.request.urlopen(
            request,
            timeout=8,
        ) as response:
            response_body = (
                response.read().decode('utf-8')
            )

            result = json.loads(
                response_body
            )

        if result.get('success') is not True:
            raise RuntimeError(
                result.get(
                    'error',
                    '충전 위치 조회 실패',
                )
            )

        data = result.get('data')

        if not isinstance(data, dict):
            raise RuntimeError(
                'API 응답에 data 객체가 없습니다.'
            )

        return {
            'x': float(data['x']),
            'y': float(data['y']),
            'theta': float(data['theta']),
        }

    def save_dock_pose_callback(
        self,
        request,
        response,
    ):
        """
        ros2 service call /save_dock_pose std_srvs/srv/Trigger "{}"
        호출 시 실행.
        """

        del request

        try:
            self.get_logger().info(
                '충전 위치 TF 평균 계산 시작'
            )

            pose = self.calculate_average_pose()

            self.get_logger().info(
                '충전 위치 평균 계산 완료: '
                f"x={pose['x']:.3f}, "
                f"y={pose['y']:.3f}, "
                f"theta={pose['theta']:.3f}"
            )

            result = self.post_dock_pose(
                pose
            )

            if result.get('success') is not True:
                response.success = False
                response.message = result.get(
                    'error',
                    '충전 위치 API 저장 실패',
                )

                return response

            response.success = True
            response.message = (
                '충전 위치 저장 완료: '
                f"x={pose['x']:.3f}, "
                f"y={pose['y']:.3f}, "
                f"theta={pose['theta']:.3f} rad"
            )

            self.get_logger().info(
                response.message
            )

        except TransformException as error:
            response.success = False
            response.message = (
                'map -> base_link TF 조회 실패: '
                f'{error}'
            )

            self.get_logger().error(
                response.message
            )

        except urllib.error.HTTPError as error:
            response.success = False
            response.message = (
                f'API HTTP 오류: {error.code}'
            )

            self.get_logger().error(
                response.message
            )

        except urllib.error.URLError as error:
            response.success = False
            response.message = (
                f'API 연결 오류: {error}'
            )

            self.get_logger().error(
                response.message
            )

        except Exception as error:
            response.success = False
            response.message = (
                f'충전 위치 저장 실패: {error}'
            )

            self.get_logger().error(
                response.message
            )

        return response

    def try_auto_initialize(self):
        """
        자율주행 시작 시 API에서 충전 위치를 조회하고
        /initialpose를 자동 발행.
        """

        if self.initialization_completed:
            return

        # AMCL이 /initialpose를 구독할 때까지 대기
        subscriber_count = (
            self.initialpose_publisher
            .get_subscription_count()
        )

        if subscriber_count == 0:
            self.get_logger().info(
                'AMCL /initialpose 구독 대기 중...'
            )

            return

        try:
            if self.charging_pose is None:
                self.charging_pose = (
                    self.get_dock_pose()
                )

                self.get_logger().info(
                    '충전 위치 조회 성공: '
                    f"x={self.charging_pose['x']:.3f}, "
                    f"y={self.charging_pose['y']:.3f}, "
                    f"theta={self.charging_pose['theta']:.3f}"
                )

            self.publish_initial_pose(
                self.charging_pose
            )

            self.initialpose_publish_count_current += 1

            self.get_logger().info(
                '/initialpose 발행: '
                f'{self.initialpose_publish_count_current}/'
                f'{self.initialpose_publish_count}'
            )

            if (
                self.initialpose_publish_count_current
                >= self.initialpose_publish_count
            ):
                self.initialization_completed = True

                if self.initialize_timer is not None:
                    self.initialize_timer.cancel()

                self.get_logger().info(
                    'AMCL 충전 위치 자동 초기화 완료'
                )

        except urllib.error.HTTPError as error:
            self.get_logger().error(
                f'충전 위치 조회 HTTP 오류: '
                f'{error.code}'
            )

        except urllib.error.URLError as error:
            self.get_logger().error(
                f'충전 위치 API 연결 오류: '
                f'{error}'
            )

        except Exception as error:
            self.get_logger().error(
                f'AMCL 자동 초기화 실패: {error}'
            )

    def publish_initial_pose(self, pose):
        """
        x, y, theta를 PoseWithCovarianceStamped로 변환하여
        /initialpose 발행.
        """

        message = PoseWithCovarianceStamped()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        message.header.frame_id = 'map'

        message.pose.pose.position.x = (
            pose['x']
        )

        message.pose.pose.position.y = (
            pose['y']
        )

        message.pose.pose.position.z = 0.0

        yaw = pose['theta']

        message.pose.pose.orientation.x = 0.0
        message.pose.pose.orientation.y = 0.0

        message.pose.pose.orientation.z = (
            math.sin(yaw / 2.0)
        )

        message.pose.pose.orientation.w = (
            math.cos(yaw / 2.0)
        )

        covariance = [0.0] * 36

        # x 분산
        covariance[0] = 0.25

        # y 분산
        covariance[7] = 0.25

        # yaw 분산
        covariance[35] = 0.0685

        message.pose.covariance = covariance

        self.initialpose_publisher.publish(
            message
        )


def main(args=None):
    rclpy.init(args=args)

    node = ChargingPoseInitializer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
