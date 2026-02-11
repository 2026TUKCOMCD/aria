from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('robot_description')
    xacro_file = os.path.join(pkg, 'urdf', 'aria_robot.urdf.xacro')

    robot_desc = xacro.process_file(xacro_file).toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        # 바퀴 joint가 continuous라 기본 0으로 유지하려고 joint_state_publisher 하나 띄움
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            output='screen'
        )
    ])
