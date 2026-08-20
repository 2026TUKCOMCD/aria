from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('kitchen_zone_name', default_value='kitchen'),

        Node(
            package='robot_bringup',
            executable='aria_cooking_ai_node',
            name='aria_cooking_ai_node',
            output='screen',
            parameters=[{
                'kitchen_zone_name': LaunchConfiguration('kitchen_zone_name'),
            }],
        ),
    ])
