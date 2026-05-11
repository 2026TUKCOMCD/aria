#!/usr/bin/env python3
import base64
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from typing import Optional
import numpy as np
import cv2
import requests
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# 구역별 고정 색상 (웹과 동일하게 유지)
ZONE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150,  50), ( 50, 200, 150), (150,  50, 200),
]


class ZoneSegmentationNode(Node):
    def __init__(self):
        super().__init__('zone_segmentation_node')

        self.declare_parameter('robot_id',            'robot_01')
        self.declare_parameter('server_url',          'http://localhost:5000')
        self.declare_parameter('min_distance',        15)
        self.declare_parameter('door_width_px',       12)
        self.declare_parameter('min_zone_area_m2',    0.15)
        self.declare_parameter('map_topic',           '/map')
        self.declare_parameter('update_interval_sec', 30.0)
        self.declare_parameter('fetch_interval_sec',  10.0)  # 이름 수신 폴링 주기

        self._robot_id    = self.get_parameter('robot_id').value
        self._server_url  = self.get_parameter('server_url').value
        self._min_dist    = self.get_parameter('min_distance').value
        self._door_w      = self.get_parameter('door_width_px').value
        self._min_area    = self.get_parameter('min_zone_area_m2').value
        self._interval    = self.get_parameter('update_interval_sec').value
        self._fetch_interval = self.get_parameter('fetch_interval_sec').value
        map_topic         = self.get_parameter('map_topic').value

        self._latest_map: Optional[OccupancyGrid] = None
        self._map_updated = False
        self._named_zones: list = []  # 사용자가 이름 붙인 zones 저장

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, 10)
        self.create_timer(self._interval,       self._process)
        self.create_timer(self._fetch_interval, self._fetch_named_zones)

        self.get_logger().info(
            f'ZoneSegmentationNode ready | robot_id={self._robot_id} '
            f'min_distance={self._min_dist}px door_width={self._door_w}px'
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

        zones, grid_img, labels = self._segment(self._latest_map)
        if not zones:
            self.get_logger().warn('No zones detected.')
            return

        self.get_logger().info(f'{len(zones)} zone(s) detected')
        map_image_b64 = self._generate_zone_image(grid_img, zones)
        self._upload(zones, map_image_b64)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment(self, msg: OccupancyGrid):
        w   = msg.info.width
        h   = msg.info.height
        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # 시각화용 grayscale 이미지 (free=255, unknown=128, occupied=0)
        grid_img = np.where(grid == 0, np.uint8(255),
                   np.where(grid == 100, np.uint8(0), np.uint8(128)))

        free = np.where(grid == 0, np.uint8(255), np.uint8(0))

        noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean   = cv2.morphologyEx(free, cv2.MORPH_OPEN, noise_k)
        dist    = ndimage.distance_transform_edt(clean)

        coords    = peak_local_max(dist, min_distance=self._min_dist, labels=clean)
        seed_mask = np.zeros(dist.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(seed_mask)
        labels     = watershed(-dist, markers, mask=clean.astype(bool))

        min_px   = self._min_area / (res ** 2)
        zone_ids = [
            int(lid) for lid in np.unique(labels)
            if lid != 0 and np.sum(labels == lid) >= min_px
        ]

        zone_ids, labels = self._merge_open_zones(labels, dist, zone_ids)

        zones = []
        for i, lid in enumerate(zone_ids):
            region   = labels == lid
            area_px  = int(np.sum(region))
            rows, cols = np.where(region)
            color    = ZONE_COLORS[i % len(ZONE_COLORS)]

            zones.append({
                'name':      f'Zone_{i + 1}',
                'center':    {'x': round(ox + cols.mean() * res, 3),
                              'y': round(oy + rows.mean() * res, 3)},
                'area':      {'x_min': round(ox + int(cols.min()) * res, 3),
                              'y_min': round(oy + int(rows.min()) * res, 3),
                              'x_max': round(ox + int(cols.max()) * res, 3),
                              'y_max': round(oy + int(rows.max()) * res, 3)},
                'color':     list(color),   # [R, G, B] — 웹과 동일한 색상
                'area_m2':   round(area_px * res ** 2, 2),
                '_mask':     region,        # 이미지 생성용 (업로드 제외)
            })

        zones.sort(key=lambda z: z['area_m2'], reverse=True)
        return zones, grid_img, labels

    def _merge_open_zones(self, labels, dist, zone_ids):
        parent = {lid: lid for lid in zone_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        zone_id_set = set(zone_ids)
        for lid in zone_ids:
            mask    = (labels == lid)
            dilated = binary_dilation(mask, iterations=1)
            border  = dilated & ~mask & (labels != 0)
            for adj in set(int(v) for v in np.unique(labels[border])
                           if v != 0 and v in zone_id_set):
                shared = border & (labels == adj)
                if shared.any() and float(dist[shared].max()) >= self._door_w:
                    union(lid, adj)

        new_labels = np.zeros_like(labels)
        for lid in zone_ids:
            new_labels[labels == lid] = find(lid)

        merged_ids = list(set(find(lid) for lid in zone_ids))
        return merged_ids, new_labels

    # ------------------------------------------------------------------
    # 구역 컬러 오버레이 이미지 생성 → base64 PNG
    # ------------------------------------------------------------------

    def _generate_zone_image(self, grid_img: np.ndarray, zones: list) -> str:
        bgr = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        for z in zones:
            color_bgr = (z['color'][2], z['color'][1], z['color'][0])  # RGB→BGR
            overlay   = bgr.copy()
            overlay[z['_mask']] = color_bgr
            bgr = cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0)

        _, buf = cv2.imencode('.png', bgr)
        return base64.b64encode(buf).decode('utf-8')

    # ------------------------------------------------------------------
    # 서버 업로드
    # ------------------------------------------------------------------

    def _upload(self, zones: list, map_image_b64: str) -> None:
        url     = f'{self._server_url}/robots/{self._robot_id}/zones'
        payload = {
            'zones': [
                {
                    'name':    z['name'],
                    'center':  z['center'],
                    'area':    z['area'],
                    'color':   z['color'],
                    'area_m2': z['area_m2'],
                }
                for z in zones
            ],
            'map_image': map_image_b64,   # base64 PNG — 웹에서 바로 표시 가능
        }
        try:
            resp = requests.put(url, json=payload, timeout=10)
            if resp.status_code == 200:
                self.get_logger().info(f'Uploaded {len(zones)} zone(s) → {url}')
            else:
                self.get_logger().error(f'Upload failed [{resp.status_code}]: {resp.text}')
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'HTTP request failed: {e}')

    # ------------------------------------------------------------------
    # 사용자가 붙인 이름 수신 (HTTP 폴링)
    # MQTT를 쓴다면 이 타이머 대신 MQTT 구독으로 교체
    # ------------------------------------------------------------------

    def _fetch_named_zones(self) -> None:
        url = f'{self._server_url}/robots/{self._robot_id}/zones'
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return

            data  = resp.json()
            zones = data.get('zones', [])

            # 이름이 실제로 붙여진 경우만 처리 (Zone_N 형태는 아직 미완료)
            named = [z for z in zones if not z.get('name', '').startswith('Zone_')]
            if not named:
                return

            if named != self._named_zones:
                self._named_zones = named
                for z in named:
                    self.get_logger().info(
                        f"[이름 수신] {z['name']} | "
                        f"center=({z['center']['x']}, {z['center']['y']})"
                    )

        except requests.exceptions.RequestException:
            pass  # 서버 미응답은 조용히 무시


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
