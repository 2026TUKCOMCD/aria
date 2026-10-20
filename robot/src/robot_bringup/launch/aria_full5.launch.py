import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    robot_bringup_share = get_package_share_directory(
        "my_robot_bringup"
    )

    autonomous_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                robot_bringup_share,
                "launch",
                "autonomous_navigation.launch.py",
            )
        )
    )

    return LaunchDescription([
        Node(
            package="my_robot_odom",
            executable="esp32_serial_bridge_node",
            name="esp32_serial_bridge_node",
            output="screen",
        ),

        autonomous_launch,

        Node(
            package="robot_bringup",
            executable="aria_mqtt_node",
            name="aria_mqtt_node",
            output="screen",
        ),

        Node(
            package="robot_bringup",
            executable="aria_controller_node",
            name="aria_controller_node",
            output="screen",
        ),

        Node(
            package="my_robot_ai_control",
            executable="data_collector",
            name="data_collector_node",
            output="screen",
        ),

        Node(
            package="my_robot_ai_control",
            executable="ai_cleaner",
            name="ai_cleaner_node",
            output="screen",
        ),
    ])
