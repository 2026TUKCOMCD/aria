import os
import json
import base64
import yaml
import requests
import time
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

def upload_map():
    """요구사항: 데이터 전송, 예외 처리, 응답 처리 통합 구현"""
    try:
        # --- [1. 데이터 전송: 변환 및 데이터 구성] ---
        if not PGM_FILE.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {PGM_FILE}")
            return

        img = read_pgm(PGM_FILE)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        with open(YAML_FILE, 'r') as f:
            meta = yaml.safe_load(f)

        payload = {
            "image_base64": img_b64,
            "map_name": f"map_{int(time.time())}",
            "metadata": {
                "resolution": meta.get('resolution', 0.05),
                "width": img.size[0],
                "height": img.size[1],
                "origin": meta.get('origin', [0.0, 0.0, 0.0])
            }
        }

        # --- [2. 예외 처리: 전송 실패 시 재시도] ---
        for i in range(1, MAX_RETRIES + 1):
            try:
                print(f"📡 서버 전송 중... ({i}/{MAX_RETRIES})")
                res = requests.post(API_URL, json=payload, timeout=15)

                # --- [3. 응답 처리: 저장 성공 여부 및 map_id 수신] ---
                if res.status_code in [200, 201]:
                    data = res.json()
                    print(f"✅ 업로드 성공! Map ID: {data.get('map_id')}")
                    return
                else:
                    print(f"⚠️ 서버 응답 에러: {res.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"❌ 네트워크 연결 실패: {e}")
            
            if i < MAX_RETRIES: time.sleep(2)

        print("최종 업로드 실패: 모든 재시도 횟수를 소진했습니다.")

    except Exception as e:
        print(f"❗ 오류 발생: {e}")

if __name__ == "__main__":
    if API_URL:
        upload_map()
    else:
        print("❌ API URL 이 설정되지 않았습니다.")
