from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 신버전 AI(my_robot_ai_control)만 단독으로 띄우는 테스트용 런치.
    # 구버전 요리AI(aria_cooking_ai_node, 카메라+YOLO)는 제거되었다 —
    # aria_controller_node가 보내는 /aria/ai_active 없이 단독 실행하면
    # ai_cleaner_node는 대기만 하고 움직이지 않는다(정상 동작).
    return LaunchDescription([
        Node(
            package='my_robot_ai_control',
            executable='data_collector',
            name='data_collector_node',
            output='screen',
        ),

        Node(
            package='my_robot_ai_control',
            executable='ai_cleaner',
            name='ai_cleaner_node',
            output='screen',
        ),
    ])
