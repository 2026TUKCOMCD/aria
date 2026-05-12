import json
from pathlib import Path

import requests

import rclpy
from rclpy.node import Node


class MapUploadNode(Node):
    def __init__(self):
        super().__init__('upload_map')

        self.declare_parameter('input_dir', '/srv/aria/users/hs/aria/robot/maps_export')
        self.declare_parameter('map_id', 'aria_map_clean')
        self.declare_parameter('version', 1)

        self.declare_parameter('robot_id', 'robot001')
        self.declare_parameter('base_url', 'http://127.0.0.1:8000')
        self.declare_parameter('bearer_token', '')
        self.declare_parameter('timeout_sec', 20.0)

        self.input_dir = self.get_parameter('input_dir').get_parameter_value().string_value
        self.map_id = self.get_parameter('map_id').get_parameter_value().string_value
        self.version = self.get_parameter('version').get_parameter_value().integer_value

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.base_url = self.get_parameter('base_url').get_parameter_value().string_value.rstrip('/')
        self.bearer_token = self.get_parameter('bearer_token').get_parameter_value().string_value
        self.timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value

        self.run_upload()

    def run_upload(self):
        image_path = Path(self.input_dir) / f'{self.map_id}_v{self.version}.png'
        json_path = Path(self.input_dir) / f'{self.map_id}_v{self.version}.json'

        if not image_path.exists():
            self.get_logger().error(f'Image file not found: {image_path}')
            rclpy.shutdown()
            return

        if not json_path.exists():
            self.get_logger().error(f'JSON file not found: {json_path}')
            rclpy.shutdown()
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            metadata_full = json.load(f)

        # API 스펙에 맞춰 필요한 필드만 추려서 metadata 구성
        metadata_payload = {
            'resolution': metadata_full['resolution'],
            'origin': [
                metadata_full['origin']['x'],
                metadata_full['origin']['y'],
                metadata_full['origin']['yaw']
            ],
            'width': metadata_full['width'],
            'height': metadata_full['height']
        }

        url = f'{self.base_url}/robots/{self.robot_id}/map'

        headers = {}
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'

        self.get_logger().info(f'Uploading to: {url}')
        self.get_logger().info(f'Image: {image_path}')
        self.get_logger().info(f'Metadata: {metadata_payload}')

        with open(image_path, 'rb') as image_file:
            files = {
                'map_image': (image_path.name, image_file, 'image/png')
            }

            data = {
                'metadata': json.dumps(metadata_payload, ensure_ascii=False)
            }

            try:
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout_sec
                )
            except requests.RequestException as e:
                self.get_logger().error(f'Upload failed: {e}')
                rclpy.shutdown()
                return

        self.get_logger().info(f'Status code: {response.status_code}')
        self.get_logger().info(f'Response: {response.text}')

        if response.status_code == 201:
            self.get_logger().info('Map upload succeeded. (201 Created)')
        else:
            self.get_logger().error('Map upload failed.')

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MapUploadNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
