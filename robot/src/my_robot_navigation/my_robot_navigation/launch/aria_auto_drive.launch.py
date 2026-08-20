import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 패키지 경로 추출
    ydlidar_share = get_package_share_directory('ydlidar_ros2_driver')
    nav2_share = get_package_share_directory('nav2_bringup')
    
    # 2. 라이다 구동 노드
    ydlidar_node = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_ros2_driver_node',
        output='screen',
        parameters=[
            os.path.join(ydlidar_share, 'params', 'X2.yaml'),
            {'frame_id': 'laser_frame', 'inverted': True, 'scan_qos_reliability': 'best_effort'}
        ]
    )

    # 3. 오도메트리 노드 (ESP32 Bridge)
    odom_node = Node(
        package='my_robot_odom',
        executable='esp32_wheel_odom_node',
        name='esp32_wheel_odom_node',
        output='screen'
    )

    # 4. TF 트리 (라이다 위치)
    tf_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['-0.04', '0.0', '0.440', '0.0', '0.0', '0.0', 'base_link', 'laser_frame']
    )

    # 5. TF 트리 (ToF 위치)
    tf_tof = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.13', '0.0', '0.01', '0.0', '0.0', '0.0', 'base_link', 'tof_frame']
    )

    # 6. SLAM (지도 작성)
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            os.path.expanduser('~/mapper_params.yaml'),
            {'use_sim_time': False}
        ]
    )

    # 7. NAV2 내비게이션 런치 연동
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_share, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': '/srv/aria/users/hs/ros2_last/src/my_robot_navigation/my_robot_navigation/config/nav2_params.yaml'
        }.items()
    )

    # 8. 자동 주행 AI (explore_lite)
    explore_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('explore_lite'), 'launch', 'explore.launch.py'
        ))
    )

    return LaunchDescription([
        ydlidar_node,
        odom_node,
        tf_laser,
        tf_tof,
        slam_node,
        nav2_launch,
        explore_launch
    ])
