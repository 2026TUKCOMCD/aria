import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_yaml = LaunchConfiguration('map_yaml')

    my_robot_bringup_dir = get_package_share_directory('my_robot_bringup')
    amcl_config = os.path.join(my_robot_bringup_dir, 'config', 'amcl.yaml')
    lifecycle_config = os.path.join(my_robot_bringup_dir, 'config', 'localization_lifecycle.yaml')
    ekf_config = '/srv/aria/users/hs/ros2_last/src/my_robot_odom/config/ekf.yaml'

    # aria_controller_node의 LOCALIZE 프로필: AMCL + navigate_to_pose 만 묶어서 띄운다.
    # SLAM(mapping.launch.py)과는 동시에 뜨면 안 되므로(map->odom TF 충돌),
    # 항상 aria_controller_node가 이 두 launch를 서브프로세스로 배타적으로 스왑한다.
    nav2_config = '/srv/aria/users/hs/ros2_last/src/my_robot_navigation/my_robot_navigation/config/nav2_params.yaml'
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    # esp32_wheel_odom_node(Tier1)는 odom TF를 직접 발행하지 않는다(코드에서 주석 처리됨) —
    # /odom + /imu/data를 융합해 odom->base_link TF를 발행하는 건 이 EKF의 몫이다.
    # mapping.launch.py와 동일 설정을 재사용하며, MAP 프로필과 동시에 뜨지 않으므로 충돌 없다.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config],
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': False}],
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_config],
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[lifecycle_config],
    )

    navigate_to_pose = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'navigation_launch.py')
        ),
        launch_arguments={'use_sim_time': 'False', 'params_file': nav2_config}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_yaml',
            default_value='/srv/aria/users/hs/ros2_last/maps/my_final_map.yaml',
        ),
        ekf_node,
        map_server_node,
        amcl_node,
        lifecycle_manager_node,
        navigate_to_pose,
    ])
