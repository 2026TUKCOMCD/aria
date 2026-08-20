import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    nav_dir = get_package_share_directory('my_robot_navigation')
    param_file = os.path.join(nav_dir, 'config', 'nav2_params.yaml')
    
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')

    nav2_with_slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            'slam': 'True', # 정적 맵 대신 실시간 slam_toolbox 백그라운드 바인딩
            'params_file': param_file,
            'use_sim_time': 'false',
            'autostart': 'true'
        }.items()
    )

    esp32_bridge_node = Node(
        package='my_robot_navigation',
        executable='esp32_bridge_node.py',
        name='esp32_bridge_node',
        output='screen'
    )

    explore_lite_node = Node(
        package='explore_lite',
        executable='explore_node',
        name='explore_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'track_unknown_space': True,
            'min_frontier_size': 0.5,
            'potential_scale': 1.0,
            'gain_scale': 1.0,
            'transform_tolerance': 0.3
        }]
    )

    return LaunchDescription([nav2_with_slam_node, esp32_bridge_node, explore_lite_node])
