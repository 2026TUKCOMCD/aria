import json
import math
import os
from datetime import datetime, timezone, timedelta

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid


class OccupancyToPngJsonNode(Node):
    def __init__(self):
        super().__init__('occupancy_to_png_json')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('output_dir', '/srv/aria/users/hs/aria/robot/maps_export')
        self.declare_parameter('map_id', 'aria_map')
        self.declare_parameter('version', 1)
        self.declare_parameter('save_once', True)
        self.declare_parameter('flip_y_for_image', True)

        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.map_id = self.get_parameter('map_id').get_parameter_value().string_value
        self.version = self.get_parameter('version').get_parameter_value().integer_value
        self.save_once = self.get_parameter('save_once').get_parameter_value().bool_value
        self.flip_y_for_image = self.get_parameter('flip_y_for_image').get_parameter_value().bool_value

        os.makedirs(self.output_dir, exist_ok=True)

        self.saved = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            qos_profile
        )

        self.get_logger().info(f'Subscribed to: {self.map_topic}')
        self.get_logger().info(f'Output dir: {self.output_dir}')

    def quaternion_to_yaw(self, z: float, w: float) -> float:
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info('map_callback triggered')
        self.get_logger().info(f'map size: {msg.info.width} x {msg.info.height}')


        if self.save_once and self.saved:
            return

        width = msg.info.width
        height = msg.info.height
        resolution = float(msg.info.resolution)

        if width == 0 or height == 0:
            self.get_logger().warn('Received empty map.')
            return

        data = np.array(msg.data, dtype=np.int16).reshape((height, width))

        rgb = np.zeros((height, width, 3), dtype=np.uint8)

        rgb[data == -1] = [128, 128, 128]   # unknown
        rgb[data == 0] = [255, 255, 255]    # free
        rgb[data == 100] = [0, 0, 0]        # occupied
        rgb[(data > 0) & (data < 100)] = [0, 0, 0]

        if self.flip_y_for_image:
            rgb = np.flipud(rgb)

        image_filename = f'{self.map_id}_v{self.version}.png'
        json_filename = f'{self.map_id}_v{self.version}.json'

        image_path = os.path.join(self.output_dir, image_filename)
        json_path = os.path.join(self.output_dir, json_filename)

        self.get_logger().info('starting save...')
        Image.fromarray(rgb).save(image_path)

        origin = msg.info.origin
        yaw = self.quaternion_to_yaw(origin.orientation.z, origin.orientation.w)

        kst = timezone(timedelta(hours=9))
        created_at = datetime.now(kst).isoformat()

        metadata = {
            'map_id': self.map_id,
            'version': int(self.version),
            'created_at': created_at,
            'frame_id': msg.header.frame_id,
            'resolution': resolution,
            'width': int(width),
            'height': int(height),
            'origin': {
                'x': float(origin.position.x),
                'y': float(origin.position.y),
                'yaw': float(yaw)
            },
            'image': image_filename,
            'coordinate_transform': {
                'world_to_pixel': {
                    'px': '(x - origin_x) / resolution',
                    'py': '(y - origin_y) / resolution',
                    'py_img': 'height - py'
                },
                'flip_y_for_image': self.flip_y_for_image
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f'Saved image: {image_path}')
        self.get_logger().info(f'Saved metadata: {json_path}')

        self.saved = True

        if self.save_once:
            self.get_logger().info('save_once=True, shutting down.')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyToPngJsonNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
