import os
import json

# 현재 파일 위치 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_lake")
MANIFEST_PATH = os.path.join(BASE_DIR, "valid_manifest.json")

def clean_data():
    valid_list = []
    print(f"🧹 [C5-1] 무결성 검사 시작: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ data_lake 폴더가 없습니다: {DATA_DIR}")
        return

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".json"): continue
            
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # [수정 포인트 1] 키 이름을 'raw_logs'로 변경
                logs = data.get('raw_logs', [])
                
                # 무결성 기준: 데이터 개수 800개 이상
                if len(logs) < 800:
                    # print(f"⚠️ {file}: 데이터 부족 ({len(logs)}개)")
                    continue 
                
                # [수정 포인트 2] logs 안의 데이터 키(pm25, voc)는 기존과 동일
                pm_vals = [d.get('pm25', 0) for d in logs]
                voc_vals = [d.get('voc', 0) for d in logs]
                
                # 이상치 필터링: PM2.5(3000), VOC(1000)
                if max(pm_vals) > 3000 or max(voc_vals) > 1000: continue 
                
                valid_list.append(file_path)
            except: continue

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_list, f, indent=4)
    
    print(f"✅ 완료! {len(valid_list)}개의 파일이 등록되었습니다.")

if __name__ == "__main__":
    clean_data()