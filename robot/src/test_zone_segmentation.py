"""
B4-1 구역 분할 알고리즘 테스트 스크립트
Distance Transform + Watershed 방식으로 구역 분할
"""
import sys
import os
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# 스크립트 위치 기준으로 maps 디렉토리 경로 설정 (robot/src/ → robot/maps/)
_MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps')

# ── 설정 ──────────────────────────────────────────────────────────────────
MAP_PGM  = os.path.join(_MAP_DIR, "map_with_doors.pgm")
MAP_YAML = os.path.join(_MAP_DIR, "map_with_doors.yaml")
OUT_PATH = os.path.join(_MAP_DIR, "zone_result.png")

RESOLUTION   = 0.02   # m/px
ORIGIN_X     = -1.94  # m
ORIGIN_Y     = -0.567 # m
FREE_THRESH  = 0.10  # 0.25→0.10: 배경 gray(픽셀=205, p≈0.196)를 free에서 제외
OCC_THRESH   = 0.65

MIN_DISTANCE     = 15   # px — 구역 씨앗 간 최소 거리
MIN_ZONE_AREA_M2 = 0.15  # m²  이하 구역 제거
DOOR_WIDTH_PX    = 12   # 경계 dist transform 값이 이 이상이면 열린 공간 → 합침
# ─────────────────────────────────────────────────────────────────────────


def pgm_to_free_mask(pgm_path: str):
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] 파일을 열 수 없습니다: {pgm_path}")
        sys.exit(1)
    p = (255.0 - img.astype(np.float32)) / 255.0
    free = np.where(p < FREE_THRESH, np.uint8(255), np.uint8(0))
    return free, img


def segment_zones_watershed(free: np.ndarray, min_dist: int, min_area_m2: float):
    # Step 1: 잡음 제거
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(free, cv2.MORPH_OPEN, noise_k)

    # Step 2: Distance Transform — 각 픽셀에서 가장 가까운 장애물까지의 거리
    dist = ndimage.distance_transform_edt(clean)

    # Step 3: Local Maxima → 각 구역의 씨앗(seed) 선정
    coords = peak_local_max(dist, min_distance=min_dist, labels=clean)
    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(seed_mask)

    # Step 4: Watershed — 거리 변환의 안장점(saddle point)에서 경계 생성
    labels = watershed(-dist, markers, mask=clean.astype(bool))

    # Step 5: 구역 정보 추출 및 소형 구역 필터링
    min_px = min_area_m2 / (RESOLUTION ** 2)
    zones = []

    for lid in np.unique(labels):
        if lid == 0:
            continue
        region = labels == lid
        area_px = int(np.sum(region))
        if area_px < min_px:
            continue

        rows, cols = np.where(region)
        col_min, col_max = int(cols.min()), int(cols.max())
        row_min, row_max = int(rows.min()), int(rows.max())

        zones.append({
            'id':          lid,
            'center':      (round(ORIGIN_X + cols.mean() * RESOLUTION, 3),
                            round(ORIGIN_Y + rows.mean() * RESOLUTION, 3)),
            'area':        (round(ORIGIN_X + col_min * RESOLUTION, 3),
                            round(ORIGIN_Y + row_min * RESOLUTION, 3),
                            round(ORIGIN_X + col_max * RESOLUTION, 3),
                            round(ORIGIN_Y + row_max * RESOLUTION, 3)),
            'area_m2':     round(area_px * RESOLUTION ** 2, 2),
            'bbox_px':     (col_min, row_min, col_max - col_min, row_max - row_min),
            'label_mask':  region,
            'centroid_px': (cols.mean(), rows.mean()),
        })

    zones.sort(key=lambda z: z['area_m2'], reverse=True)
    return zones, dist, labels


def merge_open_zones(labels: np.ndarray, dist: np.ndarray,
                     zones: list, threshold_px: float):
    """
    인접 구역 사이 경계의 distance transform 최댓값이 threshold_px 이상이면
    '열린 공간(문틈 없음)'으로 판단해 합침.
    """
    from scipy.ndimage import binary_dilation

    zone_ids = [z['id'] for z in zones]
    parent = {lid: lid for lid in zone_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for lid in zone_ids:
        mask = (labels == lid)
        dilated = binary_dilation(mask, iterations=1)
        border = dilated & ~mask & (labels != 0)
        adjacent = set(int(v) for v in np.unique(labels[border]) if v != 0 and v in zone_ids)

        for adj in adjacent:
            shared = border & (labels == adj)
            if shared.any():
                max_dist = float(dist[shared].max())
                if max_dist >= threshold_px:
                    union(lid, adj)

    # 합쳐진 구역 재구성
    new_labels = np.zeros_like(labels)
    for lid in zone_ids:
        new_labels[labels == lid] = find(lid)

    merged = {}
    for lid in zone_ids:
        root = find(lid)
        if root in merged:
            continue
        region = new_labels == root
        rows, cols = np.where(region)
        merged[root] = {
            'id':          root,
            'center':      (round(ORIGIN_X + cols.mean() * RESOLUTION, 3),
                            round(ORIGIN_Y + rows.mean() * RESOLUTION, 3)),
            'area':        (round(ORIGIN_X + int(cols.min()) * RESOLUTION, 3),
                            round(ORIGIN_Y + int(rows.min()) * RESOLUTION, 3),
                            round(ORIGIN_X + int(cols.max()) * RESOLUTION, 3),
                            round(ORIGIN_Y + int(rows.max()) * RESOLUTION, 3)),
            'area_m2':     round(int(np.sum(region)) * RESOLUTION**2, 2),
            'bbox_px':     (int(cols.min()), int(rows.min()),
                            int(cols.max()-cols.min()), int(rows.max()-rows.min())),
            'label_mask':  region,
            'centroid_px': (cols.mean(), rows.mean()),
        }

    return sorted(merged.values(), key=lambda z: z['area_m2'], reverse=True), new_labels


def print_results(zones: list):
    print("\n" + "=" * 55)
    print(f"  검출된 구역 수: {len(zones)}")
    print("=" * 55)
    for i, z in enumerate(zones):
        print(f"\n  Zone {i+1} (label_id={z['id']})")
        print(f"    넓이    : {z['area_m2']} m²")
        print(f"    중심    : x={z['center'][0]}, y={z['center'][1]}")
        print(f"    영역    : x [{z['area'][0]} ~ {z['area'][2]}]")
        print(f"             y [{z['area'][1]} ~ {z['area'][3]}]")
    print("=" * 55)


def visualize(original: np.ndarray, free: np.ndarray,
              dist: np.ndarray, labels: np.ndarray,
              zones: list, out_path: str):
    h, w = original.shape
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150,  50), ( 50, 200, 150), (150,  50, 200),
    ]

    # 1) 원본 지도 (좌상)
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    canvas[:h, :w] = orig_bgr
    cv2.putText(canvas, "1) Original Map", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 2) Distance Transform 히트맵 (우상)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    dist_color[free == 0] = 0  # 장애물 영역은 검정
    canvas[:h, w:] = dist_color
    cv2.putText(canvas, "2) Distance Transform", (w + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 3) Watershed 라벨 원본 (좌하)
    label_view = orig_bgr.copy()
    unique_labels = [l for l in np.unique(labels) if l != 0]
    for i, lid in enumerate(unique_labels):
        mask = (labels == lid).astype(np.uint8)
        overlay = label_view.copy()
        overlay[mask == 1] = COLORS[i % len(COLORS)]
        label_view = cv2.addWeighted(overlay, 0.4, label_view, 0.6, 0)
    canvas[h:, :w] = label_view
    cv2.putText(canvas, f"3) Watershed (raw, {len(unique_labels)} regions)",
                (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 4) 필터링된 최종 구역 (우하)
    zone_view = orig_bgr.copy()
    for i, z in enumerate(zones):
        c_bgr = COLORS[i % len(COLORS)]
        mask = z['label_mask'].astype(np.uint8)
        overlay = zone_view.copy()
        overlay[mask == 1] = c_bgr
        zone_view = cv2.addWeighted(overlay, 0.45, zone_view, 0.55, 0)

        col, row, bw, bh = z['bbox_px']
        cv2.rectangle(zone_view, (col, row), (col + bw, row + bh), c_bgr, 1)

        cx_px = int(z['centroid_px'][0])
        cy_px = int(z['centroid_px'][1])
        cv2.putText(zone_view, f"Z{i+1} {z['area_m2']}m2",
                    (cx_px - 20, cy_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_bgr, 1)

    canvas[h:, w:] = zone_view
    cv2.putText(canvas, f"4) Final Zones ({len(zones)})",
                (w + 10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(out_path, canvas)
    print(f"[저장] 결과 이미지 → {out_path}\n")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        MAP_PGM = os.path.abspath(sys.argv[1])
    if len(sys.argv) >= 3:
        MAP_YAML = os.path.abspath(sys.argv[2])
    if len(sys.argv) >= 4:
        MIN_DISTANCE = int(sys.argv[3])

    print(f"[설정] min_distance={MIN_DISTANCE}px ({MIN_DISTANCE * RESOLUTION:.2f}m)  "
          f"min_area={MIN_ZONE_AREA_M2}m²")

    free, original = pgm_to_free_mask(MAP_PGM)
    h, w = original.shape
    print(f"[맵]   크기 {w}×{h}px  ({w * RESOLUTION:.1f}×{h * RESOLUTION:.1f}m)")

    zones, dist, labels = segment_zones_watershed(free, MIN_DISTANCE, MIN_ZONE_AREA_M2)
    print(f"[합치기 전] {len(zones)}개 구역, DOOR_WIDTH_PX={DOOR_WIDTH_PX}px ({DOOR_WIDTH_PX*RESOLUTION*2*100:.0f}cm 이상 열린 공간은 합침)")
    zones, labels = merge_open_zones(labels, dist, zones, DOOR_WIDTH_PX)

    print_results(zones)
    visualize(original, free, dist, labels, zones, OUT_PATH)
