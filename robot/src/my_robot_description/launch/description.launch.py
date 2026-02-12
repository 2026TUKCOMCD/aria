from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import subprocess

def load_robot_description():
    pkg_share = get_package_share_directory('my_robot_description')

    # 우선순위: xacro -> urdf
    candidate_xacro = [
        os.path.join(pkg_share, 'urdf', 'my_robot.urdf.xacro'),
        os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro'),
        os.path.join(pkg_share, 'urdf', 'model.urdf.xacro'),
    ]
    candidate_urdf = [
        os.path.join(pkg_share, 'urdf', 'my_robot.urdf'),
        os.path.join(pkg_share, 'urdf', 'robot.urdf'),
        os.path.join(pkg_share, 'urdf', 'model.urdf'),
    ]

    for x in candidate_xacro:
        if os.path.exists(x):
            return subprocess.check_output(['xacro', x]).decode()

    for u in candidate_urdf:
        if os.path.exists(u):
            with open(u, 'r') as f:
                return f.read()

    raise FileNotFoundError(
        f'No URDF/XACRO found under {pkg_share}/urdf. '
        f'Put a .urdf or .urdf.xacro there.'
    )

def generate_launch_description():
    robot_description = load_robot_description()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
    ])
