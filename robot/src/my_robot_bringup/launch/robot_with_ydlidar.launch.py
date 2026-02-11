from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # URDF (xacro)
    default_model = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'my_robot.urdf.xacro'
    )

    # ydlidar params
    default_params = '/srv/aria/users/hs/ros2_ws/src/ydlidar_ros2_driver/params/X2.yaml'

    model_arg = DeclareLaunchArgument(
        'model',
        default_value=default_model,
        description='Absolute path to robot urdf.xacro'
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Absolute path to ydlidar params yaml'
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', LaunchConfiguration('model')]),
        }]
    )

    # ydlidar launch include (너가 A방법으로 TF 노드 제거해둔 버전이라고 가정)
    ydlidar_launch = os.path.join(
        get_package_share_directory('ydlidar_ros2_driver'),
        'launch',
        'ydlidar_launch.py'
    )

    ydlidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ydlidar_launch),
        launch_arguments={'params_file': LaunchConfiguration('params_file')}.items()
    )

    return LaunchDescription([
        model_arg,
        params_arg,
        robot_state_pub,
        ydlidar_include
    ])
