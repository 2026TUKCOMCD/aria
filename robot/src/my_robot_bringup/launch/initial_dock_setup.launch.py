from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # MQTT 노드는 별도 상시 실행 프로세스로 관리합니다.
        # 이 런치에는 최초 충전 위치 저장 서비스만 포함합니다.
        Node(
            package="charging_pose_initializer",
            executable="charging_pose_initializer_node",
            name="charging_pose_initializer",
            output="screen",
            parameters=[
                {
                    "base_api_url": (
                        "https://ph7ckbtbl3.execute-api."
                        "ap-northeast-2.amazonaws.com"
                    ),
                    "robot_id": 1,
                    "auto_initialize": False,
                    "sample_count": 10,
                    "sample_interval": 0.2,
                }
            ],
        ),
    ])
