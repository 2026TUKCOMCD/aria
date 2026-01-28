import json
import random
from datetime import datetime, timedelta

def augment_data(input_file, output_file, multiply=10):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            base_data = json.load(f)

        augmented_results = []
        print(f"{len(base_data)}개의 기본 시나리오를 {multiply}배로 증강합니다...")

        for session in base_data:
            for i in range(multiply):
                # 각 복사본마다 시작 시간을 i시간만큼 뒤로 미룹니다.
                time_offset = timedelta(hours=i)
                
                new_meta = session['meta'].copy()
                new_logs = []

                for log in session['logs']:
                    # [수정] ISO 형식을 자동으로 인식하도록 변경 ✅
                    orig_time = datetime.fromisoformat(log['measured_at'])
                    new_time = orig_time + time_offset

                    # 수치 미세 변형 (Jittering)
                    variation = lambda x: round(x * random.uniform(0.95, 1.05), 2)

                    new_logs.append({
                        "measured_at": new_time.isoformat(), # 저장할 때도 ISO 형식 유지
                        "temperature": variation(log['temperature']),
                        "humidity": variation(log['humidity']),
                        "pm25": variation(log['pm25']),
                        "voc": variation(log['voc'])
                    })

                augmented_results.append({
                    "meta": new_meta,
                    "logs": new_logs
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_results, f, indent=4, ensure_ascii=False)
        
        print(f"증강 완료! 총 {len(augmented_results)}개의 세션이 '{output_file}'에 저장되었습니다.")

    except Exception as e:
        print(f"증강 중 오류 발생: {e}")

if __name__ == "__main__":
    augment_data("aria_synthetic_data.json", "aria_augmented_data.json", multiply=10)