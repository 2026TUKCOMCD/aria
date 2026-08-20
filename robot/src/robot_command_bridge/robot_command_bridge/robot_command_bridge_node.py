#!/usr/bin/env python3

import json
import math
import os
import queue
import ssl
import threading
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class RobotCommandBridge(Node):
    """
    AWS IoT MQTT 이동 명령을 Nav2 NavigateToPose Action으로 변환한다.

    구독 Topic:
      aria/{robot_id}/cmd/nav

    기본 Payload:
      {
        "type": "MOVE_TO",
        "x": 12.5,
        "y": 5.0,
        "theta": 0.0
      }

    추가 지원:
      {"type": "CANCEL"}
      {"type": "STOP"}

    참고:
    - x, y는 ROS map 좌표계 기준 미터 값이어야 한다.
    - theta는 radian 단위이며 생략하면 0.0을 사용한다.
    - 방 이동과 충전 복귀 모두 웹이 최종 좌표를 MOVE_TO로 보내면 된다.
    """

    def __init__(self) -> None:
        super().__init__('robot_command_bridge')

        self.declare_parameter('robot_id', 1)

        self.declare_parameter(
            'mqtt_host',
            os.environ.get(
                'ARIA_MQTT_HOST',
                'adecukeeb0iln-ats.iot.ap-northeast-2.amazonaws.com',
            ),
        )
        self.declare_parameter(
            'mqtt_port',
            int(os.environ.get('ARIA_MQTT_PORT', '8883')),
        )
        self.declare_parameter(
            'mqtt_client_id',
            os.environ.get(
                'ARIA_MQTT_CLIENT_ID',
                'aria-robot-1-nav-bridge',
            ),
        )
        self.declare_parameter(
            'mqtt_command_topic',
            os.environ.get('ARIA_MQTT_COMMAND_TOPIC', ''),
        )
        self.declare_parameter(
            'mqtt_status_topic',
            os.environ.get('ARIA_MQTT_STATUS_TOPIC', ''),
        )

        self.declare_parameter(
            'mqtt_ca_cert',
            os.environ.get(
                'ARIA_MQTT_CA_CERT',
                '/srv/aria/users/hs/ros2_last/src/certs/AmazonRootCA1.pem',
            ),
        )
        self.declare_parameter(
            'mqtt_client_cert',
            os.environ.get(
                'ARIA_MQTT_CLIENT_CERT',
                '/srv/aria/users/hs/ros2_last/src/certs/'
                'bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d'
                '-certificate.pem.crt',
            ),
        )
        self.declare_parameter(
            'mqtt_private_key',
            os.environ.get(
                'ARIA_MQTT_PRIVATE_KEY',
                '/srv/aria/users/hs/ros2_last/src/certs/'
                'bebcdfcaa5098bff6d11a031c2924eebe96a244ea1b4e0b971a79dc8189cb21d'
                '-private.pem.key',
            ),
        )

        self.robot_id = int(self.get_parameter('robot_id').value)
        self.mqtt_host = str(self.get_parameter('mqtt_host').value)
        self.mqtt_port = int(self.get_parameter('mqtt_port').value)
        self.mqtt_client_id = str(
            self.get_parameter('mqtt_client_id').value
        )

        configured_command_topic = str(
            self.get_parameter('mqtt_command_topic').value
        )
        configured_status_topic = str(
            self.get_parameter('mqtt_status_topic').value
        )

        self.command_topic = (
            configured_command_topic
            if configured_command_topic
            else f'aria/{self.robot_id}/cmd/nav'
        )
        self.status_topic = (
            configured_status_topic
            if configured_status_topic
            else f'aria/{self.robot_id}/status/nav'
        )

        self.ca_cert = str(self.get_parameter('mqtt_ca_cert').value)
        self.client_cert = str(
            self.get_parameter('mqtt_client_cert').value
        )
        self.private_key = str(
            self.get_parameter('mqtt_private_key').value
        )

        self.navigate_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose',
        )

        self.current_goal_handle = None
        self.goal_lock = threading.Lock()
        self.command_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        self.mqtt_client = mqtt.Client(
            client_id=self.mqtt_client_id,
            protocol=mqtt.MQTTv311,
        )
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect

        self.configure_tls()
        self.start_mqtt()

        # MQTT callback thread에서 ROS Action을 직접 호출하지 않고
        # ROS executor thread에서 큐를 처리한다.
        self.command_timer = self.create_timer(
            0.1,
            self.process_command_queue,
        )

    def configure_tls(self) -> None:
        required_files = {
            'CA': self.ca_cert,
            'certificate': self.client_cert,
            'private key': self.private_key,
        }

        missing = [
            f'{name}: {path}'
            for name, path in required_files.items()
            if not path or not os.path.isfile(path)
        ]

        if missing:
            raise FileNotFoundError(
                'MQTT 인증서 파일을 찾을 수 없습니다: '
                + ', '.join(missing)
            )

        self.mqtt_client.tls_set(
            ca_certs=self.ca_cert,
            certfile=self.client_cert,
            keyfile=self.private_key,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )

    def start_mqtt(self) -> None:
        self.mqtt_client.connect_async(
            self.mqtt_host,
            self.mqtt_port,
            keepalive=60,
        )
        self.mqtt_client.loop_start()

        self.get_logger().info(
            f'MQTT 연결 시도: {self.mqtt_host}:{self.mqtt_port}'
        )
        self.get_logger().info(
            f'MQTT 이동 명령 Topic: {self.command_topic}'
        )

    def on_mqtt_connect(
        self,
        client,
        userdata,
        flags,
        result_code,
    ) -> None:
        del userdata, flags

        if result_code != 0:
            self.get_logger().error(
                f'MQTT 연결 실패: result_code={result_code}'
            )
            return

        client.subscribe(
            self.command_topic,
            qos=1,
        )

        self.get_logger().info(
            f'MQTT 연결 성공, 구독 시작: {self.command_topic}'
        )
        self.publish_status(
            'NAV_BRIDGE_READY',
            {},
        )

    def on_mqtt_disconnect(
        self,
        client,
        userdata,
        result_code,
    ) -> None:
        del client, userdata

        self.get_logger().warning(
            f'MQTT 연결 종료: result_code={result_code}'
        )

    def on_mqtt_message(
        self,
        client,
        userdata,
        message,
    ) -> None:
        del client, userdata

        try:
            payload = json.loads(
                message.payload.decode('utf-8')
            )

            if not isinstance(payload, dict):
                raise ValueError('Payload must be a JSON object.')

            self.command_queue.put(payload)

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            self.get_logger().error(
                f'MQTT Payload 오류: {error}'
            )
            self.publish_status(
                'COMMAND_REJECTED',
                {'reason': str(error)},
            )

    def process_command_queue(self) -> None:
        while True:
            try:
                payload = self.command_queue.get_nowait()
            except queue.Empty:
                return

            try:
                self.handle_command(payload)
            except Exception as error:
                self.get_logger().error(
                    f'이동 명령 처리 실패: {error}'
                )
                self.publish_status(
                    'COMMAND_REJECTED',
                    {'reason': str(error)},
                )

    def handle_command(self, payload: Dict[str, Any]) -> None:
        command_type = str(
            payload.get('type', '')
        ).strip().upper()

        if command_type == 'MOVE_TO':
            self.handle_move_to(payload)
            return

        if command_type in ('CANCEL', 'STOP'):
            self.cancel_navigation()
            return

        raise ValueError(
            f'지원하지 않는 type입니다: {command_type}'
        )

    def handle_move_to(self, payload: Dict[str, Any]) -> None:
        if 'x' not in payload or 'y' not in payload:
            raise ValueError(
                'MOVE_TO Payload에는 x와 y가 필요합니다.'
            )

        x = float(payload['x'])
        y = float(payload['y'])
        theta = float(payload.get('theta', 0.0))

        if not all(math.isfinite(value) for value in (x, y, theta)):
            raise ValueError(
                'x, y, theta는 유한한 숫자여야 합니다.'
            )

        metadata = {
            'type': 'MOVE_TO',
            'x': x,
            'y': y,
            'theta': theta,
        }

        # 웹에서 디버깅용 값을 추가해 보내도 이동에는 영향 없음
        for key in (
            'zone_id',
            'zone_name',
            'target_type',
            'target_name',
            'map_id',
        ):
            if key in payload:
                metadata[key] = payload[key]

        self.send_navigation_goal(
            x=x,
            y=y,
            theta=theta,
            metadata=metadata,
        )

    def send_navigation_goal(
        self,
        x: float,
        y: float,
        theta: float,
        metadata: Dict[str, Any],
    ) -> None:
        if not self.navigate_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.publish_status(
                'NAVIGATION_FAILED',
                {
                    **metadata,
                    'reason': (
                        'NavigateToPose Action 서버를 '
                        '찾을 수 없습니다.'
                    ),
                },
            )
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = (
            math.sin(theta / 2.0)
        )
        goal.pose.pose.orientation.w = (
            math.cos(theta / 2.0)
        )

        future = self.navigate_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result_future: self.on_goal_response(
                result_future,
                metadata,
            )
        )

        self.publish_status(
            'NAVIGATION_REQUESTED',
            metadata,
        )

    def on_goal_response(
        self,
        future,
        metadata: Dict[str, Any],
    ) -> None:
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.publish_status(
                'NAVIGATION_REJECTED',
                metadata,
            )
            return

        with self.goal_lock:
            self.current_goal_handle = goal_handle

        self.publish_status(
            'NAVIGATION_STARTED',
            metadata,
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda final_future: self.on_navigation_result(
                final_future,
                metadata,
            )
        )

    def on_navigation_result(
        self,
        future,
        metadata: Dict[str, Any],
    ) -> None:
        result_wrapper = future.result()
        action_status = (
            result_wrapper.status
            if result_wrapper is not None
            else -1
        )

        with self.goal_lock:
            self.current_goal_handle = None

        # action_msgs/msg/GoalStatus
        # 4: SUCCEEDED, 5: CANCELED, 6: ABORTED
        if action_status == 4:
            event = 'NAVIGATION_SUCCEEDED'
        elif action_status == 5:
            event = 'NAVIGATION_CANCELED'
        else:
            event = 'NAVIGATION_FAILED'

        self.publish_status(
            event,
            {
                **metadata,
                'action_status': action_status,
            },
        )

    def cancel_navigation(self) -> None:
        with self.goal_lock:
            goal_handle = self.current_goal_handle

        if goal_handle is None:
            self.publish_status(
                'NAVIGATION_CANCEL_SKIPPED',
                {'reason': '진행 중인 이동 Goal이 없습니다.'},
            )
            return

        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda _: self.publish_status(
                'NAVIGATION_CANCEL_REQUESTED',
                {},
            )
        )

    def publish_status(
        self,
        event: str,
        data: Dict[str, Any],
    ) -> None:
        payload = {
            'event': event,
            'robot_id': self.robot_id,
            'data': data,
        }

        try:
            self.mqtt_client.publish(
                self.status_topic,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                qos=1,
            )
        except Exception as error:
            self.get_logger().error(
                f'MQTT 상태 발행 실패: {error}'
            )

    def destroy_node(self) -> bool:
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RobotCommandBridge] = None

    try:
        node = RobotCommandBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(
                f'robot_command_bridge 시작 실패: {error}'
            )
        else:
            print(
                f'robot_command_bridge 시작 실패: {error}'
            )
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
