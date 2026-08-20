from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_id   = LaunchConfiguration('robot_id')
    # aria_controller_node/air_purify_scheduler_node는 robot_id를 문자열로 선언한다.
    # LaunchConfiguration을 파라미터 dict에 그대로 넣으면 launch가 임시 params
    # yaml에 따옴표 없이 써서 "1"이 정수로 오인되어 InvalidParameterTypeException이
    # 난다 (charging_pose_initializer는 robot_id를 int로 선언해서 우연히 안 걸림).
    robot_id_str = ParameterValue(robot_id, value_type=str)

    # ── Tier 1: 항상 켜짐, 절대 재시작 안 함 ────────────────────
    # AMCL<->SLAM(Tier2)은 여기 포함하지 않는다 — aria_controller_node가
    # 시작 시점에 LOCALIZE 프로필을 `ros2 launch`서브프로세스로 직접 띄우고,
    # 매핑모드 진입/종료 시에만 MAP 프로필로 스왑한다.

    odom_node = Node(
        package='my_robot_odom',
        executable='esp32_wheel_odom_node',
        name='esp32_wheel_odom_node',
        output='screen',
    )

    bridge_node = Node(
        package='my_robot_odom',
        executable='esp32_serial_bridge_node',
        name='esp32_serial_bridge_node',
        output='screen',
        parameters=[{'port': '/dev/serial0'}],
    )

    # 라이더 구동 (mapping.launch.py와 동일 파라미터 — 이 워크스페이스에서
    # 검증된 설정을 그대로 재사용)
    ydlidar_node = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_ros2_driver_node',
        output='screen',
        parameters=[
            '/srv/aria/users/hs/ros2_last/src/ydlidar_ros2_driver/params/X2.yaml',
            {
                'port': '/dev/ttyUSB0',
                'frame_id': 'laser_frame',
                'inverted': True,
            },
        ],
    )

    laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_tf',
        arguments=['-0.04', '0.0', '0.440', '0.0', '0.0', '0.0', 'base_link', 'laser_frame'],
    )

    tof_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tof_frame_tf',
        arguments=['0.13', '0.0', '0.01', '0.0', '0.0', '0.0', 'base_link', 'tof_frame'],
    )

    charging_pose_initializer_node = Node(
        package='charging_pose_initializer',
        executable='charging_pose_initializer_node',
        name='charging_pose_initializer',
        output='screen',
        parameters=[{'robot_id': robot_id}],
    )

    mqtt_node = Node(
        package='robot_bringup',
        executable='aria_mqtt_node',
        name='aria_mqtt_node',
        output='screen',
    )

    controller_node = Node(
        package='robot_bringup',
        executable='aria_controller_node',
        name='aria_controller_node',
        output='screen',
        parameters=[{
            'robot_id':   robot_id_str,
            # server_url 강제 주입 제거 (파이썬 코드에서 .env를 읽도록 유도)
        }],
    )

    cooking_ai_node = Node(
        package='robot_bringup',
        executable='aria_cooking_ai_node',
        name='aria_cooking_ai_node',
        output='screen',
        parameters=[{
            'kitchen_zone_name': 'kitchen',
        }],
    )

    scheduler_node = Node(
        package='robot_bringup',
        executable='air_purify_scheduler_node',
        name='air_purify_scheduler_node',
        output='screen',
        parameters=[{
            'robot_id':   robot_id_str,
            # server_url 강제 주입 제거 (파이썬 코드에서 .env를 읽도록 유도)
        }],
    )

    return LaunchDescription([
        # ARIA 상시구동 진입점: Tier1 노드 전체를 한 번에 올린다.
        DeclareLaunchArgument('robot_id',   default_value='1'),
        # server_url DeclareLaunchArgument 완전히 제거

        odom_node,
        bridge_node,
        ydlidar_node,
        laser_tf_node,
        tof_tf_node,
        charging_pose_initializer_node,
        mqtt_node,
        controller_node,
        cooking_ai_node,
        scheduler_node,
    ])
