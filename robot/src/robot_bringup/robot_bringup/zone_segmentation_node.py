#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from typing import Optional
import numpy as np
import cv2
import requests


class ZoneSegmentationNode(Node):
    def __init__(self):
        super().__init__('zone_segmentation_node')

        self.declare_parameter('robot_id', 'robot_01')
        self.declare_parameter('server_url', 'http://localhost:5000')
        # 침식 커널 크기 (픽셀 단위). 해상도 0.05m 기준 16px ≈ 문틀 너비 0.8m
        self.declare_parameter('erosion_kernel_size', 16)
        self.declare_parameter('min_zone_area_m2', 1.0)
        self.declare_parameter('map_topic', '/map')
        # 맵 구독 후 이 주기(초)마다 세그멘테이션 실행
        self.declare_parameter('update_interval_sec', 30.0)

        self._robot_id: str = self.get_parameter('robot_id').value
        self._server_url: str = self.get_parameter('server_url').value
        self._erosion_k: int = self.get_parameter('erosion_kernel_size').value
        self._min_area_m2: float = self.get_parameter('min_zone_area_m2').value
        self._interval: float = self.get_parameter('update_interval_sec').value
        map_topic: str = self.get_parameter('map_topic').value

        self._latest_map: Optional[OccupancyGrid] = None
        self._map_updated: bool = False

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, 10)
        self.create_timer(self._interval, self._process)

        self.get_logger().info(
            f'ZoneSegmentationNode ready | robot_id={self._robot_id} '
            f'interval={self._interval}s erosion_k={self._erosion_k}px'
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._latest_map = msg
        self._map_updated = True

    def _process(self) -> None:
        if not self._map_updated or self._latest_map is None:
            return
        self._map_updated = False

        zones = self._segment(self._latest_map)
        if not zones:
            self.get_logger().warn(
                'No zones detected. '
                'Map may be too small or erosion_kernel_size too large.'
            )
            return

        self.get_logger().info(f'{len(zones)} zone(s) detected')
        self._upload(zones)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment(self, msg: OccupancyGrid) -> list:
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        # OccupancyGrid 데이터 → numpy 이미지
        # free(0)=255(흰색), occupied(100)/unknown(-1)=0(검정)
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        free = np.where(grid == 0, np.uint8(255), np.uint8(0))

        # Step 1: 잡음 제거 (작은 고립 픽셀 제거)
        noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(free, cv2.MORPH_OPEN, noise_k)

        # Step 2: 문틀 너비만큼 침식 → 방과 방의 연결 끊기
        door_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._erosion_k, self._erosion_k)
        )
        eroded = cv2.erode(clean, door_k)

        # Step 3: Connected Components Labeling
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )

        min_px = self._min_area_m2 / (res ** 2)
        zones = []

        for lid in range(1, n_labels):  # 0 = background
            area_px = int(stats[lid, cv2.CC_STAT_AREA])
            if area_px < min_px:
                continue

            col = int(stats[lid, cv2.CC_STAT_LEFT])
            row = int(stats[lid, cv2.CC_STAT_TOP])
            bw = int(stats[lid, cv2.CC_STAT_WIDTH])
            bh = int(stats[lid, cv2.CC_STAT_HEIGHT])

            # 픽셀 좌표 → 월드 좌표
            # OccupancyGrid: origin은 픽셀(0,0)의 월드 좌표, row 증가 = y 증가
            x_min = ox + col * res
            y_min = oy + row * res
            x_max = ox + (col + bw) * res
            y_max = oy + (row + bh) * res
            cx = ox + centroids[lid][0] * res
            cy = oy + centroids[lid][1] * res

            zones.append({
                'name': f'Zone_{lid}',
                'center': {'x': round(cx, 3), 'y': round(cy, 3)},
                'area': {
                    'x_min': round(x_min, 3),
                    'y_min': round(y_min, 3),
                    'x_max': round(x_max, 3),
                    'y_max': round(y_max, 3),
                },
                '_area_m2': round(area_px * res ** 2, 2),
            })

        # 넓은 구역부터 정렬 (Zone_1이 가장 넓은 방 = 거실)
        zones.sort(key=lambda z: z['_area_m2'], reverse=True)
        return zones

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def _upload(self, zones: list) -> None:
        url = f'{self._server_url}/robots/{self._robot_id}/zones'
        payload = {
            'zones': [
                {'name': z['name'], 'center': z['center'], 'area': z['area']}
                for z in zones
            ]
        }
        try:
            resp = requests.put(url, json=payload, timeout=10)
            if resp.status_code == 200:
                self.get_logger().info(f'Uploaded {len(zones)} zone(s) → {url}')
            else:
                self.get_logger().error(
                    f'Upload failed [{resp.status_code}]: {resp.text}'
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'HTTP request failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ZoneSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
