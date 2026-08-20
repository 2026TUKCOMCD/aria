#!/usr/bin/env python3
import os
import json
import base64
import yaml
import requests
import time
import datetime
import argparse
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ==============================
# 기본 경로 / 환경변수 설정
# ==============================

# 이 파일은 ros2_last/src/upload_map_with_zones.py 에 위치하지만,
# .env(ARIA_API_URL 등)는 별도의 aria 프로젝트 루트에 있고 저장된 맵은
# ros2_last/maps 에 있다 — 서로 다른 두 트리라 __file__ 상대경로로는
# 유도할 수 없으므로 다른 ARIA 노드들과 동일한 관례(ARIA_PROJECT_ROOT/
# ARIA_MAPS_DIR 환경변수, 기본값은 배포 경로)를 그대로 따른다.
PROJECT_ROOT = Path(os.environ.get("ARIA_PROJECT_ROOT", "/srv/aria/users/hs/aria"))
MAPS_DIR = Path(os.environ.get("ARIA_MAPS_DIR", "/srv/aria/users/hs/ros2_last/maps"))
MAPS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

API_BASE_URL = os.environ.get("ARIA_API_URL")
ROBOT_ID = os.environ.get("ROBOT_ID", "1")

MAX_RETRIES = 3


# ==============================
# 구역 분할 파라미터
# ==============================

# ROS map 기준:
# 흰색 255 = free
# 검정 0 = occupied
# 회색 205 = unknown
#
# 여기서는 확실한 흰색 영역만 free로 판단
FREE_PIXEL_THRESH = 250

MIN_DISTANCE = 12          # px, 구역 씨앗 간 최소 거리
MIN_ZONE_AREA_M2 = 0.2    # m², 너무 작은 구역 제거


# ==============================
# 파일 / YAML 처리
# ==============================

def resolve_map_files(map_name: str):
    """
    사용 예:
      python3 upload_map_with_zones.py last
      python3 upload_map_with_zones.py last.pgm
      python3 upload_map_with_zones.py /srv/.../maps/last.pgm
    """
    p = Path(map_name)

    if not p.is_absolute():
        p = MAPS_DIR / p

    if p.suffix == ".pgm":
        pgm_path = p
        yaml_path = p.with_suffix(".yaml")
    elif p.suffix == ".yaml":
        yaml_path = p
        pgm_path = p.with_suffix(".pgm")
    else:
        pgm_path = p.with_suffix(".pgm")
        yaml_path = p.with_suffix(".yaml")

    if not pgm_path.exists():
        raise FileNotFoundError(f"PGM 파일이 없습니다: {pgm_path}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 파일이 없습니다: {yaml_path}")

    return pgm_path, yaml_path


def load_map_yaml(yaml_path: Path):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    resolution = meta.get("resolution")
    origin = meta.get("origin")

    if resolution is None:
        raise ValueError("YAML에 resolution 값이 없습니다.")

    if origin is None or len(origin) < 3:
        raise ValueError("YAML에 origin 값이 없거나 형식이 잘못되었습니다.")

    return meta, float(resolution), origin


# ==============================
# 좌표 변환
# ==============================

def pixel_to_world(col, row, resolution, origin, image_height):
    """
    이미지 좌표:
      col: 왼쪽 -> 오른쪽
      row: 위 -> 아래

    ROS map world 좌표:
      origin은 보통 이미지 좌하단 기준.

    따라서 y축은 image_height - row로 뒤집어줌.
    """
    world_x = origin[0] + float(col) * resolution
    world_y = origin[1] + float(image_height - row) * resolution

    return round(world_x, 3), round(world_y, 3)


def get_polygon_from_mask(mask, resolution, origin, image_height):
    """
    구역 mask에서 polygon 외곽선 추출.
    DB에는 world 좌표 polygon으로 저장.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    polygon = []
    for col, row in contour:
        wx, wy = pixel_to_world(col, row, resolution, origin, image_height)
        polygon.append([wx, wy])

    return polygon


# ==============================
# 구역 분할
# ==============================

def run_segmentation(pgm_path: Path, resolution, origin):
    """
    PGM 맵에서 free space를 추출하고
    Distance Transform + Watershed로 구역 분할.
    """
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"PGM 이미지를 읽을 수 없습니다: {pgm_path}")

    h, w = img.shape[:2]

    # 확실한 흰색 영역만 free로 사용
    free = np.where(img >= FREE_PIXEL_THRESH, np.uint8(255), np.uint8(0))

    # 작은 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(free, cv2.MORPH_OPEN, kernel)

    # Distance Transform
    dist = ndimage.distance_transform_edt(clean)

    # 각 방/공간의 seed 찾기
    coords = peak_local_max(
        dist,
        min_distance=MIN_DISTANCE,
        labels=clean
    )

    if len(coords) == 0:
        return [], img, dist, np.zeros_like(img, dtype=np.int32)

    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True

    markers, _ = ndimage.label(seed_mask)

    labels = watershed(
        -dist,
        markers,
        mask=clean.astype(bool)
    )

    min_px = MIN_ZONE_AREA_M2 / (resolution ** 2)

    zones = []
    unique_ids = [int(l) for l in np.unique(labels) if l != 0]

    for lid in unique_ids:
        region = labels == lid
        area_px = int(np.sum(region))

        if area_px < min_px:
            continue

        rows, cols = np.where(region)

        cx_px = float(cols.mean())
        cy_px = float(rows.mean())

        center_x, center_y = pixel_to_world(
            cx_px,
            cy_px,
            resolution,
            origin,
            h
        )

        x_min, y_max = pixel_to_world(
            cols.min(),
            rows.min(),
            resolution,
            origin,
            h
        )

        x_max, y_min = pixel_to_world(
            cols.max(),
            rows.max(),
            resolution,
            origin,
            h
        )

        polygon = get_polygon_from_mask(region, resolution, origin, h)

        zones.append({
            "id": lid,
            "name": f"Zone {lid}",
            "center": {
                "x": center_x,
                "y": center_y
            },
            "area": {
                "x_min": round(x_min, 3),
                "y_min": round(y_min, 3),
                "x_max": round(x_max, 3),
                "y_max": round(y_max, 3)
            },
            "polygon": polygon,

            # 내부 시각화용 필드
            "_label_mask": region.astype(np.uint8),
            "_centroid_px": (cx_px, cy_px),
            "_area_m2": round(area_px * (resolution ** 2), 2),
            "_bbox_px": (
                int(cols.min()),
                int(rows.min()),
                int(cols.max() - cols.min()),
                int(rows.max() - rows.min())
            )
        })

    # 큰 구역부터 정렬
    zones.sort(key=lambda z: z["_area_m2"], reverse=True)

    # 정렬 후 id/name을 보기 좋게 다시 부여
    for idx, z in enumerate(zones, start=1):
        z["id"] = idx
        z["name"] = f"Zone {idx}"

    return zones, img, dist, labels


# ==============================
# Zone Preview 이미지 생성
# ==============================

def build_zone_preview_image(original: np.ndarray, zones: list):
    """
    원본 흑백 맵 위에 구역별 색상 overlay만 입힌 PNG 생성.
    숫자/텍스트/사각형 없음.
    """
    if len(original.shape) == 2:
        zone_view = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        zone_view = original.copy()

    COLORS = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
        (200, 150, 50),
        (50, 200, 150),
        (150, 50, 200),
        (80, 180, 255),
        (180, 80, 255),
        (255, 180, 80),
    ]

    for i, z in enumerate(zones):
        color = COLORS[i % len(COLORS)]
        mask = z["_label_mask"].astype(bool)

        overlay = zone_view.copy()
        overlay[mask] = color
        zone_view = cv2.addWeighted(overlay, 0.45, zone_view, 0.55, 0)

    return zone_view

def image_to_base64_png(img: np.ndarray):
    ok, buffer = cv2.imencode(".png", img)

    if not ok:
        raise ValueError("PNG 인코딩 실패")

    return base64.b64encode(buffer).decode("utf-8")


def sanitize_zones_for_payload(zones: list):
    """
    numpy 배열 같은 내부 필드는 JSON 전송 불가하므로 제거.
    Lambda에는 DB 저장용 정보만 보낸다.
    """
    clean = []

    for z in zones:
        clean.append({
            "id": z["id"],
            "name": z["name"],
            "center": z["center"],
            "area": z["area"],
            "polygon": z["polygon"]
        })

    return clean


# ==============================
# API URL
# ==============================

def build_endpoint_url():
    if not API_BASE_URL:
        raise ValueError(".env에 ARIA_API_URL이 설정되어 있지 않습니다.")

    base = API_BASE_URL.rstrip("/")

    if "robot_id=" in base:
        raise ValueError(
            "ARIA_API_URL에 robot_id=1 형태가 들어가 있습니다. "
            "ARIA_API_URL은 API Gateway 기본 URL만 넣고, ROBOT_ID=1로 따로 설정하세요."
        )

    # 이미 전체 endpoint를 넣은 경우:
    # https://.../robots/1/map
    if base.endswith("/map"):
        return base

    return f"{base}/robots/{ROBOT_ID}/map"


# ==============================
# 업로드
# ==============================

def upload_map(map_name: str, no_zones: bool = False):
    pgm_path, yaml_path = resolve_map_files(map_name)

    print("====================================")
    print("🗺️ 맵 업로드 시작")
    print(f"PGM  : {pgm_path}")
    print(f"YAML : {yaml_path}")
    print(f"API_BASE_URL : {API_BASE_URL}")
    print(f"ROBOT_ID : {ROBOT_ID}")
    print("====================================")

    meta, resolution, origin = load_map_yaml(yaml_path)

    original_img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        raise ValueError(f"맵 이미지를 읽을 수 없습니다: {pgm_path}")

    height, width = original_img.shape[:2]

    if no_zones:
        zones = []
        zones_payload = []
        print("⚠️ zones 없이 맵만 업로드합니다.")

        # 구역 없이 업로드하면 흑백 맵 그대로 BGR PNG 변환
        preview_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    else:
        print("🧠 구역 분할 중...")
        zones, _, _, _ = run_segmentation(pgm_path, resolution, origin)
        print(f"📍 검출된 구역 수: {len(zones)}개")

        preview_img = build_zone_preview_image(original_img, zones)
        zones_payload = sanitize_zones_for_payload(zones)

    # 로컬에도 preview 저장
    preview_path = MAPS_DIR / f"{Path(pgm_path).stem}_zone_preview.png"
    cv2.imwrite(str(preview_path), preview_img)
    print(f"🖼️ S3 업로드용 zone preview 저장: {preview_path}")

    # S3에 올라갈 이미지는 색칠된 preview PNG
    img_b64 = image_to_base64_png(preview_img)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "image_base64": img_b64,
        "map_name": f"{Path(pgm_path).stem}_{current_time}",
        "metadata": {
            "resolution": resolution,
            "width": width,
            "height": height,
            "origin": origin
        },
        "zones": zones_payload
    }

    endpoint_url = build_endpoint_url()

    headers = {
        "Content-Type": "application/json"
    }

    for i in range(1, MAX_RETRIES + 1):
        try:
            print(f"📡 업로드 요청 ({i}/{MAX_RETRIES})")
            print(f"URL: {endpoint_url}")

            res = requests.post(
                endpoint_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            print(f"HTTP Status: {res.status_code}")

            if res.status_code in [200, 201]:
                try:
                    data = res.json()
                except Exception:
                    data = {"raw_response": res.text}

                print("✅ 업로드 성공")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return True

            print("⚠️ 서버 응답:")
            print(res.text)

        except requests.exceptions.RequestException as e:
            print(f"❌ 네트워크 오류: {e}")

        time.sleep(2)

    print("❌ 최종 업로드 실패")
    return False


# ==============================
# main
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="ARIA map uploader with colored zone preview"
    )

    parser.add_argument(
        "map",
        nargs="?",
        default="last",
        help="업로드할 맵 이름. 예: last, aria_real_map, /path/to/map.pgm"
    )

    parser.add_argument(
        "--no-zones",
        action="store_true",
        help="구역 분할 없이 맵 이미지만 업로드"
    )

    args = parser.parse_args()

    upload_map(args.map, no_zones=args.no_zones)


if __name__ == "__main__":
    main()
