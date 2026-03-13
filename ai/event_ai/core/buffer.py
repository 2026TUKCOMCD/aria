import numpy as np
from collections import deque
from datetime import datetime
import json

class AirQualityBuffer:
    def __init__(self, max_len=900):
        self.buffer = deque(maxlen=max_len)

    def add_data(self, temp, humi, pm25, voc):
        data_point = {
            'measured_at': datetime.now().isoformat(),
            'temperature': float(temp),
            'humidity': float(humi),
            'pm25': float(pm25),
            'voc': float(voc)
        }
        self.buffer.append(data_point)

    def get_full_logs(self):
        return list(self.buffer)

    def get_session_features(self):
        """AI 추론 및 1단계 트리거를 위한 모든 특징 계산"""
        if len(self.buffer) < 10: return None

        pm25_vals = [d['pm25'] for d in self.buffer]
        voc_vals = [d['voc'] for d in self.buffer]
        x = np.arange(len(pm25_vals))

        # 1. 기울기(Slope) 계산
        pm25_slope = np.polyfit(x, pm25_vals, 1)[0]
        voc_slope = np.polyfit(x, voc_vals, 1)[0]

        # 2. 최신 절대 수치 추출
        current_pm25 = pm25_vals[-1]
        current_voc = voc_vals[-1]

        return {
            'pm25_slope': float(pm25_slope),
            'voc_slope': float(voc_slope),
            'current_pm25': float(current_pm25),
            'current_voc': float(current_voc),
            'pm25_std': float(np.std(pm25_vals)),
            'voc_std': float(np.std(voc_vals)),
            'pm25_range': float(np.max(pm25_vals) - np.min(pm25_vals))
        }

    def make_package(self, robot_id, predicted_prob, yolo_verified, features):
        return {
            "meta": {
                "robot_id": str(robot_id),
                "timestamp": datetime.now().isoformat(),
                "predicted_prob": round(float(predicted_prob), 4),
                "yolo_verified": bool(yolo_verified),
                "stats": features
            },
            "raw_logs": list(self.buffer)
        }

    def clear(self):
        self.buffer.clear()