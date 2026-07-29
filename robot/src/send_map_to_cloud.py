import os
import json
import base64
import yaml
import requests
import time
import datetime
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv

# 1. 환경 변수 로드 (.env 탐색)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 경로 및 설정값
MAPS_DIR = project_root / 'robot' / 'maps'
PGM_FILE = MAPS_DIR / 'aria_real_map.pgm'
YAML_FILE = MAPS_DIR / 'aria_real_map.yaml'
API_URL = os.environ.get("ARIA_API_URL")
MAX_RETRIES = 3 # 예외 처리: 최대 3회 재시도

def read_pgm(filename):
    """PGM 형식을 지원하기 위한 읽기 함수 (P2/P5 통합)"""
    with open(filename, 'rb') as f:
        header = f.readline().decode().strip()
        def get_next():
            l = f.readline().decode().strip()
            while not l or l.startswith('#'): l = f.readline().decode().strip()
            return l
        w, h = map(int, get_next().split())
        max_val = int(get_next())
        data = f.read() if header == 'P5' else bytes(map(int, f.read().split()))
        return Image.frombytes('L', (w, h), data)

def validate_metadata(meta):
    """[적용 사항 2] 이슈 #230: 지도 메타데이터 무결성 검증"""
    required_keys = ['resolution', 'origin']
    for key in required_keys:
        if key not in meta or meta[key] is None:
            print(f"❌ 데이터 오류: 필수 메타데이터 '{key}' 가 누락되었습니다.")
            return False
    
    # 원점(origin) 데이터가 리스트 형태이고 3개의 요소(x, y, yaw)를 가졌는지 확인
    if not isinstance(meta.get('origin'), list) or len(meta.get('origin')) < 3:
        print("❌ 데이터 오류: origin 데이터 형식이 올바르지 않습니다.")
        return False
        
    return True

def upload_map():
    """요구사항: 데이터 전송, 예외 처리, 응답 처리 및 히스토리 관리 구현"""
    try:
        # --- [1. 데이터 전송 준비] ---
        if not PGM_FILE.exists() or not YAML_FILE.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {PGM_FILE} 또는 {YAML_FILE}")
            return

        # 이미지 읽기 및 Base64 변환
        img = read_pgm(PGM_FILE)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # YAML 메타데이터 읽기
        with open(YAML_FILE, 'r') as f:
            meta = yaml.safe_load(f)

        # [적용 사항 2] 메타데이터 검증 실행
        if not validate_metadata(meta):
            print("❗ 메타데이터 검증 실패로 업로드를 중단합니다.")
            return

        # [적용 사항 1] 이슈 #230: 타임스탬프를 이용한 버전 관리 명명 규칙 적용
        # 형식: aria_map_20260512_2030 (연월일_시분)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        map_version_name = f"aria_map_{current_time}"

        payload = {
            "image_base64": img_b64,
            "map_name": map_version_name,
            "metadata": {
                "resolution": meta.get('resolution'),
                "width": img.size[0],
                "height": img.size[1],
                "origin": meta.get('origin')
            }
        }

        # --- [2. 예외 처리: 전송 실패 시 재시도] ---
        for i in range(1, MAX_RETRIES + 1):
            try:
                print(f"📡 서버 전송 중... ({map_version_name}) ({i}/{MAX_RETRIES})")
                res = requests.post(API_URL, json=payload, timeout=15)

                # --- [3. 응답 처리: 저장 성공 여부 및 map_id 수신] ---
                if res.status_code in [200, 201]:
                    data = res.json()
                    print(f"✅ 업로드 성공 ! Map ID : {data.get('map_id')}")
                    return
                else:
                    print(f"⚠️ 서버 응답 에러 : {res.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"❌ 네트워크 연결 실패 : {e}")

            if i < MAX_RETRIES: time.sleep(2)

        print("최종 업로드 실패 : 모든 재시도 횟수를 소진했습니다.")

    except Exception as e:
        print(f"❗ 오류 발생 : {e}")

if __name__ == "__main__":
    if API_URL:
        upload_map()
    else:
        print("❌ API URL 이 설정되지 않았습니다.")
