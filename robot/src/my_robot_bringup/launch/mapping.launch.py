from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ekf_config = '/srv/aria/users/hs/ros2_last/src/my_robot_odom/config/ekf.yaml'
    nav2_config = '/srv/aria/users/hs/ros2_last/src/my_robot_navigation/my_robot_navigation/config/nav2_params.yaml'
    slam_config = '/home/hs/mapper_params.yaml'

    return_to_init = LaunchConfiguration('return_to_init')

    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    explore_launch_dir = os.path.join(get_package_share_directory('explore_lite'), 'launch')

    # odom/ydlidar/정적 TF는 Tier1(aria_full.launch.py)에서 항상 떠있으므로
    # 여기서 중복으로 띄우지 않는다 — 이 launch는 aria_controller_node가
    # MAP 프로필로 스왑할 때만 켜지는 SLAM 전용 스택이다.
    nodes = []

    # EKF 노드를 2초 지연 실행 (오도메트리 데이터가 먼저 들어오게 함)
    nodes.append(TimerAction(period=2.0, actions=[
        Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', parameters=[ekf_config])
    ]))

    # SLAM 노드를 4초 지연 실행
    nodes.append(TimerAction(period=4.0, actions=[
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
             parameters=[slam_config, {'use_sim_time': False, 'throttle_scans': 3}])
    ]))

    # Nav2를 6초 지연 실행 (SLAM이 준비된 후 실행)
    nodes.append(TimerAction(period=6.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
            launch_arguments={'use_sim_time': 'False', 'params_file': nav2_config}.items()),
    ]))

    # Explore는 Nav2보다 4초 늦게(10초 지연) 실행한다 — 이전엔 Nav2와 같은
    # 시점(6초)에 같이 띄웠더니 라즈베리파이 CPU 4코어가 전부 포화되어
    # bt_navigator가 액션서버 연결 1초 타임아웃 안에 응답을 못 받고
    # 활성화에 실패하는 문제가 실측 재현됨. Nav2 라이프사이클 활성화가
    # 먼저 끝나고 안정된 뒤에 explore_lite를 띄워 부하를 분산시킨다.
    # 맵 저장/업로드는 aria_controller_node가 explore_lite의 explore/status
    # 토픽에서 EXPLORATION_COMPLETE를 받은 뒤 직접 처리하므로(OnShutdown에 더
    # 이상 의존하지 않음), return_to_init은 기본적으로 꺼서(false) explore_lite
    # 자체 귀환과 controller의 귀환 이동이 겹치지 않게 한다.
    nodes.append(TimerAction(period=10.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(explore_launch_dir, 'explore.launch.py')),
            launch_arguments={'return_to_init': return_to_init}.items())
    ]))

    return LaunchDescription([
        DeclareLaunchArgument('return_to_init', default_value='false'),
        *nodes,
    ])
