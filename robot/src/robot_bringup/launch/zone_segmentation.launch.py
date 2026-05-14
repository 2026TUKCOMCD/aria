from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_id',             default_value='robot_01'),
        DeclareLaunchArgument('server_url',           default_value='http://localhost:5000'),
        DeclareLaunchArgument('min_distance',         default_value='15'),   # Watershed 씨앗 간 최소 거리 (px)
        DeclareLaunchArgument('door_width_px',        default_value='12'),   # 열린 공간 판단 기준 (px)
        DeclareLaunchArgument('min_zone_area_m2',     default_value='0.15'),
        DeclareLaunchArgument('map_topic',            default_value='/map'),
        DeclareLaunchArgument('update_interval_sec',  default_value='30.0'),
        DeclareLaunchArgument('fetch_interval_sec',   default_value='10.0'),  # 이름 수신 폴링 주기

        Node(
            package='robot_bringup',
            executable='zone_segmentation_node',
            name='zone_segmentation_node',
            output='screen',
            parameters=[{
                'robot_id':            LaunchConfiguration('robot_id'),
                'server_url':          LaunchConfiguration('server_url'),
                'min_distance':        LaunchConfiguration('min_distance'),
                'door_width_px':       LaunchConfiguration('door_width_px'),
                'min_zone_area_m2':    LaunchConfiguration('min_zone_area_m2'),
                'map_topic':           LaunchConfiguration('map_topic'),
                'update_interval_sec': LaunchConfiguration('update_interval_sec'),
                'fetch_interval_sec':  LaunchConfiguration('fetch_interval_sec'),
            }],
        ),
    ])
