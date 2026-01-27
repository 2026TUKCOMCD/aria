import paho.mqtt.client as mqtt
import psycopg2
import os
import json
from datetime import datetime

# 환경 변수
DB_HOST = os.getenv('DB_HOST', 'db')
DB_NAME = os.getenv('DB_NAME', 'aria')
DB_USER = os.getenv('DB_USER', 'user')
DB_PASS = os.getenv('DB_PASSWORD')
MQTT_BROKER = os.getenv('MQTT_BROKER', 'mqtt-broker')

def save_to_db(data):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()

        # 1. sensor_sessions 테이블에 모든 분석 데이터 넣기 (UPSERT 방식)
        session_query = """
        INSERT INTO sensor_sessions (
            session_id, predicted_prob, yolo_verified, final_label, 
            pm25_slope, temp_hum_corr, pm_voc_corr, 
            pm25_std, voc_std, pm25_range
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE SET
            predicted_prob = EXCLUDED.predicted_prob,
            yolo_verified = EXCLUDED.yolo_verified,
            final_label = EXCLUDED.final_label,
            pm25_slope = EXCLUDED.pm25_slope,
            temp_hum_corr = EXCLUDED.temp_hum_corr,
            pm_voc_corr = EXCLUDED.pm_voc_corr,
            pm25_std = EXCLUDED.pm25_std,
            voc_std = EXCLUDED.voc_std,
            pm25_range = EXCLUDED.pm25_range;
        """
        cur.execute(session_query, (
            data.get('session_id'), data.get('predicted_prob'), data.get('yolo_verified'),
            data.get('final_label'), data.get('pm25_slope'), data.get('temp_hum_corr'),
            data.get('pm_voc_corr'), data.get('pm25_std'), data.get('voc_std'), data.get('pm25_range')
        ))

        # 2. sensor_data_logs 테이블에 시계열 로그 넣기
        log_query = """
        INSERT INTO sensor_data_logs (session_id, temperature, humidity, pm25, voc, measured_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(log_query, (
            data.get('session_id'),
            data.get('temp'), data.get('hum'),
            data.get('pm25'), data.get('voc'),
            datetime.now()
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"[{datetime.now()}] >>> 세션 및 로그 통합 저장 완료!", flush=True)
    except Exception as e:
        print(f"!!! 저장 실패: {e}", flush=True)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        save_to_db(payload)
    except Exception as e:
        print(f"메시지 처리 에러: {e}", flush=True)

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message

client.connect(MQTT_BROKER, 1883, 60)
client.subscribe("aria/sensor/data")
print("ARIA Subscriber 통합 모드 가동 중...", flush=True)
client.loop_forever()