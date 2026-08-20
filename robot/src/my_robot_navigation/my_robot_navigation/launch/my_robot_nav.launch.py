import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    nav_dir = get_package_share_directory('my_robot_navigation')
    
    # 맵과 파라미터 파일 경로 
    map_file = os.path.join(nav_dir, 'maps', 'aria_real_map.yaml')
    param_file = os.path.join(nav_dir, 'config', 'nav2_params.yaml')
    
    # Nav2 기본 bringup 런치 파일 가져오기
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')

    # 실행 환경 구성
    nav2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            'map': map_file,
            'params_file': param_file,
            'use_sim_time': 'false', # 실제 로봇 구동이므로 false
            'autostart': 'true'
        }.items()
    )

    # 하드웨어 통신 브릿지 노드 추가 (신경망)
    esp32_bridge_node = Node(
        package='my_robot_navigation',
        executable='esp32_bridge_node.py',
        name='esp32_bridge_node',
        output='screen'
    )

    # 두 노드를 한 바구니에 담아서 동시에 리턴!
    return LaunchDescription([
        nav2_node,
        esp32_bridge_node,
        static_tf_node
    ])
