from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_id',             default_value='robot_01'),
        DeclareLaunchArgument('server_url',           default_value='http://localhost:5000'),
        # 문틀 너비 기준 침식 크기 (px). 해상도 0.05m 기준 16px ≈ 0.8m
        DeclareLaunchArgument('erosion_kernel_size',  default_value='16'),
        DeclareLaunchArgument('min_zone_area_m2',     default_value='1.0'),
        DeclareLaunchArgument('map_topic',            default_value='/map'),
        # 맵 안정화 후 처리 주기 (초)
        DeclareLaunchArgument('update_interval_sec',  default_value='30.0'),

        Node(
            package='robot_bringup',
            executable='zone_segmentation_node',
            name='zone_segmentation_node',
            output='screen',
            parameters=[{
                'robot_id':            LaunchConfiguration('robot_id'),
                'server_url':          LaunchConfiguration('server_url'),
                'erosion_kernel_size': LaunchConfiguration('erosion_kernel_size'),
                'min_zone_area_m2':    LaunchConfiguration('min_zone_area_m2'),
                'map_topic':           LaunchConfiguration('map_topic'),
                'update_interval_sec': LaunchConfiguration('update_interval_sec'),
            }],
        ),
    ])
