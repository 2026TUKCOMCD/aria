import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav2_bringup_share = get_package_share_directory('nav2_bringup')
    ydlidar_share = get_package_share_directory('ydlidar_ros2_driver')

    ekf_params = (
        '/srv/aria/users/hs/ros2_last/src/'
        'my_robot_odom/config/ekf.yaml'
    )

    lidar_params = os.path.join(
        ydlidar_share,
        'params',
        'X2.yaml',
    )

    map_file = LaunchConfiguration('map')
    nav2_params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_id = LaunchConfiguration('robot_id')
    base_api_url = LaunchConfiguration('base_api_url')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=(
                '/srv/aria/users/hs/ros2_last/maps/'
                'my_final_map.yaml'
            ),
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=(
                '/srv/aria/users/hs/ros2_last/src/'
                'my_robot_navigation/my_robot_navigation/'
                'config/nav2_params.yaml'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'robot_id',
            default_value='1',
        ),
        DeclareLaunchArgument(
            'base_api_url',
            default_value=(
                'https://ph7ckbtbl3.execute-api.'
                'ap-northeast-2.amazonaws.com'
            ),
        ),

        # 1. 휠 오도메트리
        Node(
            package='my_robot_odom',
            executable='esp32_wheel_odom_node',
            name='esp32_wheel_odom_node',
            output='screen',
        ),

        # 2. EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            arguments=[
                '--ros-args',
                '--params-file',
                ekf_params,
            ],
        ),

        # 3. YDLIDAR
        Node(
            package='ydlidar_ros2_driver',
            executable='ydlidar_ros2_driver_node',
            name='ydlidar_ros2_driver_node',
            output='screen',
            arguments=[
                '--ros-args',
                '--params-file',
                lidar_params,
                '-p',
                'frame_id:=laser_frame',
                '-p',
                'inverted:=true',
            ],
        ),

        # 4. base_link -> laser_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_static_tf_publisher',
            output='screen',
            arguments=[
                '--x', '-0.04',
                '--y', '0.0',
                '--z', '0.440',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'laser_frame',
            ],
        ),

        # 5. base_link -> tof_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tof_static_tf_publisher',
            output='screen',
            arguments=[
                '--x', '0.13',
                '--y', '0.0',
                '--z', '0.01',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'tof_frame',
            ],
        ),

        # 6. 저장된 지도 기반 Nav2 + AMCL
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    nav2_bringup_share,
                    'launch',
                    'bringup_launch.py',
                )
            ),
            launch_arguments={
                'map': map_file,
                'use_sim_time': use_sim_time,
                'params_file': nav2_params_file,
            }.items(),
        ),

        # 7. DB 충전 위치로 AMCL initialpose 자동 설정
        Node(
            package='charging_pose_initializer',
            executable='charging_pose_initializer_node',
            name='charging_pose_initializer',
            output='screen',
            parameters=[
                {
                    'base_api_url': base_api_url,
                    'robot_id': robot_id,
                    'auto_initialize': True,
                    'sample_count': 10,
                    'sample_interval': 0.2,
                    'initialpose_publish_count': 3,
                    'retry_interval': 2.0,
                }
            ],
        ),

        # 8. aria/1/cmd/nav의 MOVE_TO 명령을 NavigateToPose로 변환
 #       Node(
  #          package='robot_command_bridge',
   #         executable='robot_command_bridge_node',
    #        name='robot_command_bridge',
     #       output='screen',
      #      parameters=[
       #         {
#                    'robot_id': robot_id,
        #        }
         #   ],
       # ),
    ])
