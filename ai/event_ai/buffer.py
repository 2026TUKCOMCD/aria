import numpy as np
from collections import deque
from datetime import datetime

class AirQualityBuffer:
    def __init__(self, max_len=900):
        # 30분치(2초 주기 시 900개) 데이터를 담는 큐
        self.buffer = deque(maxlen=max_len)

    def add_data(self, temp, humi, pm25, voc):
        data_point = {
            'measured_at': datetime.now().isoformat(), # ISO8601 형식
            'temperature': float(temp),
            'humidity': float(humi),
            'pm25': float(pm25),
            'voc': float(voc)
        }
        self.buffer.append(data_point)

    def get_full_logs(self):
        # 현재 버퍼에 쌓인 900개 로그 전체 반환 (클라우드 전송용)
        return list(self.buffer)

    def get_session_features(self):
        # AI 추론 및 DB 메타데이터용 통계량 계산
        if len(self.buffer) < 10: return None

        pm25_vals = [d['pm25'] for d in self.buffer]
        voc_vals = [d['voc'] for d in self.buffer]

        # 1. 기울기(Slope) 계산
        x = np.arange(len(pm25_vals))
        pm25_slope = np.polyfit(x, pm25_vals, 1)[0]

        return {
            'pm25_slope': float(pm25_slope),
            'pm25_std': float(np.std(pm25_vals)),
            'voc_std': float(np.std(voc_vals)),
            'pm25_range': float(np.max(pm25_vals) - np.min(pm25_vals))
        }

    def clear(self):
        self.buffer.clear()