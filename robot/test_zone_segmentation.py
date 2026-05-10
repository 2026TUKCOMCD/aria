"""
B4-1 구역 분할 알고리즘 테스트 스크립트
zone_segmentation_node.py 와 동일한 알고리즘을 PGM 파일에 직접 적용해
결과를 시각화하고 구역별 정보를 출력한다.
"""
import sys
import cv2
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────────────
MAP_PGM  = "maps/aria_map_clean.pgm"
MAP_YAML = "maps/aria_map_clean.yaml"
OUT_PATH = "maps/zone_result.png"

RESOLUTION   = 0.02   # m/px  (yaml 기준)
ORIGIN_X     = -1.94  # m
ORIGIN_Y     = -0.567 # m
FREE_THRESH  = 0.25   # nav2 map_server 기준
OCC_THRESH   = 0.65

EROSION_KERNEL_SIZE = 40   # px  (~0.8m 문틀 너비, 0.02m/px 기준)
MIN_ZONE_AREA_M2    = 1.0  # m²  이하 구역 제거
# ─────────────────────────────────────────────────────────────────────────

def pgm_to_free_mask(pgm_path: str) -> np.ndarray:
    """
    nav2 map_server trinary 모드와 동일한 방식으로 PGM → free 마스크 변환
    negate=0 기준:
      p = (255 - pixel) / 255
      p < free_thresh  → free   (255)
      p > occ_thresh   → occupied (0)
      else             → unknown  (0)
    """
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] 파일을 열 수 없습니다: {pgm_path}")
        sys.exit(1)

    p = (255.0 - img.astype(np.float32)) / 255.0
    free = np.where(p < FREE_THRESH, np.uint8(255), np.uint8(0))
    return free, img.shape


def segment_zones(free: np.ndarray, erosion_k: int, min_area_m2: float):
    # Step 1: 잡음 제거
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(free, cv2.MORPH_OPEN, noise_k)

    # Step 2: 침식 → 문틀 연결 끊기
    door_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_k, erosion_k)
    )
    eroded = cv2.erode(clean, door_k)

    # Step 3: Connected Components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8
    )

    min_px = min_area_m2 / (RESOLUTION ** 2)
    zones = []

    for lid in range(1, n_labels):
        area_px = int(stats[lid, cv2.CC_STAT_AREA])
        if area_px < min_px:
            continue

        col = int(stats[lid, cv2.CC_STAT_LEFT])
        row = int(stats[lid, cv2.CC_STAT_TOP])
        bw  = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh  = int(stats[lid, cv2.CC_STAT_HEIGHT])

        x_min = ORIGIN_X + col * RESOLUTION
        y_min = ORIGIN_Y + row * RESOLUTION
        x_max = ORIGIN_X + (col + bw) * RESOLUTION
        y_max = ORIGIN_Y + (row + bh) * RESOLUTION
        cx    = ORIGIN_X + centroids[lid][0] * RESOLUTION
        cy    = ORIGIN_Y + centroids[lid][1] * RESOLUTION

        zones.append({
            'id':       lid,
            'center':   (round(cx, 3), round(cy, 3)),
            'area':     (round(x_min,3), round(y_min,3),
                         round(x_max,3), round(y_max,3)),
            'area_m2':  round(area_px * RESOLUTION**2, 2),
            'bbox_px':  (col, row, bw, bh),
            'label_mask': labels == lid,
        })

    zones.sort(key=lambda z: z['area_m2'], reverse=True)
    return zones, eroded


def visualize(original_pgm: np.ndarray, free: np.ndarray,
              eroded: np.ndarray, zones: list, out_path: str):
    h, w = original_pgm.shape

    # 4분할 캔버스: 원본 | free mask | 침식 결과 | 구역 색상
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # 1) 원본 지도 (좌상)
    orig_bgr = cv2.cvtColor(original_pgm, cv2.COLOR_GRAY2BGR)
    canvas[:h, :w] = orig_bgr
    cv2.putText(canvas, "1) Original Map", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 2) Free mask (우상)
    free_bgr = cv2.cvtColor(free, cv2.COLOR_GRAY2BGR)
    canvas[:h, w:] = free_bgr
    cv2.putText(canvas, "2) Free Mask", (w + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 3) 침식 결과 (좌하)
    eroded_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    canvas[h:, :w] = eroded_bgr
    cv2.putText(canvas, "3) After Erosion (door gaps)", (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 4) 구역 색상 (우하) — 원본 지도 위에 오버레이
    zone_view = orig_bgr.copy()
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
    ]
    for i, z in enumerate(zones):
        col_bgr = colors[i % len(colors)]
        mask = z['label_mask'].astype(np.uint8)
        # 반투명 채우기
        overlay = zone_view.copy()
        overlay[mask == 1] = col_bgr
        zone_view = cv2.addWeighted(overlay, 0.45, zone_view, 0.55, 0)

        # 바운딩 박스
        c, r, bw, bh = z['bbox_px']
        cv2.rectangle(zone_view, (c, r), (c + bw, r + bh), col_bgr, 1)

        # 라벨
        label = f"Z{i+1} {z['area_m2']}m2"
        cx_px = int(centroids_from_zone(z))
        cy_px = int(r + bh // 2)
        cv2.putText(zone_view, label, (c + 4, r + bh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col_bgr, 1)

    canvas[h:, w:] = zone_view
    cv2.putText(canvas, f"4) Detected Zones ({len(zones)})",
                (w + 10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(out_path, canvas)
    print(f"\n[저장] 결과 이미지 → {out_path}")


def centroids_from_zone(z):
    col, row, bw, bh = z['bbox_px']
    return col + bw / 2


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


if __name__ == '__main__':
    print(f"[설정] erosion_kernel={EROSION_KERNEL_SIZE}px  "
          f"min_area={MIN_ZONE_AREA_M2}m²")

    free, shape = pgm_to_free_mask(MAP_PGM)
    print(f"[맵]   크기 {shape[1]}×{shape[0]}px  "
          f"({shape[1]*RESOLUTION:.1f}×{shape[0]*RESOLUTION:.1f}m)")

    original = cv2.imread(MAP_PGM, cv2.IMREAD_GRAYSCALE)

    # --- 침식 결과도 시각화용으로 가져오기 위해 분리 ---
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean   = cv2.morphologyEx(free, cv2.MORPH_OPEN, noise_k)
    door_k  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE)
    )
    eroded  = cv2.erode(clean, door_k)

    # centroids 접근을 위해 다시 계산
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8
    )

    # zones 재구성 (centroids 포함)
    min_px = MIN_ZONE_AREA_M2 / (RESOLUTION ** 2)
    zones = []
    for lid in range(1, n_labels):
        area_px = int(stats[lid, cv2.CC_STAT_AREA])
        if area_px < min_px:
            continue
        col = int(stats[lid, cv2.CC_STAT_LEFT])
        row = int(stats[lid, cv2.CC_STAT_TOP])
        bw  = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh  = int(stats[lid, cv2.CC_STAT_HEIGHT])
        x_min = ORIGIN_X + col * RESOLUTION
        y_min = ORIGIN_Y + row * RESOLUTION
        x_max = ORIGIN_X + (col + bw) * RESOLUTION
        y_max = ORIGIN_Y + (row + bh) * RESOLUTION
        cx    = ORIGIN_X + centroids[lid][0] * RESOLUTION
        cy    = ORIGIN_Y + centroids[lid][1] * RESOLUTION
        zones.append({
            'id':        lid,
            'center':    (round(cx, 3), round(cy, 3)),
            'area':      (round(x_min,3), round(y_min,3),
                          round(x_max,3), round(y_max,3)),
            'area_m2':   round(area_px * RESOLUTION**2, 2),
            'bbox_px':   (col, row, bw, bh),
            'label_mask': labels == lid,
            'centroid_px': (centroids[lid][0], centroids[lid][1]),
        })
    zones.sort(key=lambda z: z['area_m2'], reverse=True)

    print_results(zones)

    # 시각화 (centroids 재사용)
    h, w = original.shape
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    orig_bgr  = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    free_bgr  = cv2.cvtColor(free,     cv2.COLOR_GRAY2BGR)
    eroded_bgr = cv2.cvtColor(eroded,  cv2.COLOR_GRAY2BGR)

    canvas[:h, :w]  = orig_bgr
    canvas[:h, w:]  = free_bgr
    canvas[h:, :w]  = eroded_bgr

    cv2.putText(canvas, "1) Original Map",          (10,    20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, "2) Free Mask",             (w+10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, "3) After Erosion",         (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, f"4) Zones ({len(zones)})", (w+10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    zone_view = orig_bgr.copy()
    COLORS = [
        (255,100,100),(100,255,100),(100,100,255),
        (255,255,100),(255,100,255),(100,255,255),
        (200,150, 50),( 50,200,150),(150, 50,200),
    ]
    for i, z in enumerate(zones):
        c_bgr = COLORS[i % len(COLORS)]
        mask  = z['label_mask'].astype(np.uint8)
        overlay = zone_view.copy()
        overlay[mask == 1] = c_bgr
        zone_view = cv2.addWeighted(overlay, 0.45, zone_view, 0.55, 0)

        col, row, bw, bh = z['bbox_px']
        cv2.rectangle(zone_view, (col, row), (col+bw, row+bh), c_bgr, 1)

        label = f"Z{i+1} {z['area_m2']}m2"
        cx_px = int(z['centroid_px'][0])
        cy_px = int(z['centroid_px'][1])
        cv2.putText(zone_view, label, (cx_px - 20, cy_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_bgr, 1)

    canvas[h:, w:] = zone_view
    cv2.imwrite(OUT_PATH, canvas)
    print(f"[저장] 결과 이미지 → {OUT_PATH}\n")
