import numpy as np
from collections import deque
from datetime import datetime
import json

class AirQualityBuffer:
    def __init__(self, max_len=900):
        # 30분치(2초 주기 시 900개) 데이터를 담는 큐
        self.buffer = deque(maxlen=max_len)

    def add_data(self, temp, humi, pm25, voc):
        """센서로부터 전달받은 단일 데이터를 버퍼에 추가합니다."""
        data_point = {
            'measured_at': datetime.now().isoformat(), # ISO8601 형식
            'temperature': float(temp),
            'humidity': float(humi),
            'pm25': float(pm25),
            'voc': float(voc)
        }
        self.buffer.append(data_point)

    def get_full_logs(self):
        """현재 버퍼에 쌓인 로그 전체를 리스트로 반환합니다."""
        return list(self.buffer)

    def get_session_features(self):
        """AI 추론 및 메타데이터 생성을 위한 통계적 특징을 계산합니다."""
        if len(self.buffer) < 10: return None

        pm25_vals = [d['pm25'] for d in self.buffer]
        voc_vals = [d['voc'] for d in self.buffer]

        # 1. 기울기(Slope) 계산: 시간에 따른 미세먼지 농도 변화량
        x = np.arange(len(pm25_vals))
        pm25_slope = np.polyfit(x, pm25_vals, 1)[0]

        return {
            'pm25_slope': float(pm25_slope),
            'pm25_std': float(np.std(pm25_vals)),
            'voc_std': float(np.std(voc_vals)),
            'pm25_range': float(np.max(pm25_vals) - np.min(pm25_vals))
        }

    # --- Feature 4-2: 데이터 원샷 패키징 (One-shot Packaging) 추가 ---
    def make_package(self, robot_id, predicted_prob, yolo_verified, features):
        """
        [30분 센서 데이터] + [AI 판단 확률] + [YOLO 검증 결과]를 하나의 JSON 객체로 조립합니다.
        기존 DB 테이블(sensor_sessions, sensor_data_logs) 구조를 통합한 비정규화 형태입니다.
        """
        return {
            "meta": {
                "robot_id": str(robot_id),
                "timestamp": datetime.now().isoformat(),
                "predicted_prob": round(float(predicted_prob), 4),
                "yolo_verified": bool(yolo_verified),
                "stats": features # pm25_slope, pm25_std 등이 포함됨
            },
            "raw_logs": list(self.buffer) # 30분 상세 시계열 로그 전체
        }

    def clear(self):
        """세션 종료 후 버퍼를 비웁니다."""
        self.buffer.clear()