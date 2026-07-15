import json
import psycopg2
import os
import urllib3

# ==========================================
# 1. 환경 변수 설정 (Lambda Configuration)
# ==========================================
# DB 연결 설정
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD') or os.environ.get('DB_PASS')
DB_PORT = os.environ.get('DB_PORT', "5432")

# EC2 웹 서버 설정
EC2_IP = os.environ.get('EC2_IP')

# HTTP 요청 관리자 (EC2 전송용)
http = urllib3.PoolManager()

def make_connection():
    """PostgreSQL 데이터베이스 연결 객체 생성"""
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5
        )
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

def lambda_handler(event, context):
    print("LWT 비정상 종료 이벤트 수신:", event)
    
    # IoT Core Rule에서 SQL로 추출한 데이터 파싱
    # (주의: IoT Rule SQL에서 SELECT *, topic(2) as robot_id 형태로 넘겨주어야 함)
    robot_id = event.get('robot_id', 'unknown_robot')
    status = event.get('status', 'Offline')
    reason = event.get('reason', 'Connection Lost')
    
    conn = None
    cursor = None

    try:
        # ==========================================
        # 2. DB 업데이트 로직 (오프라인 상태 기록)
        # ==========================================
        conn = make_connection()
        if conn:
            cursor = conn.cursor()
            
            # 비정상 종료 기록이므로 센서값은 기본값(0)으로 처리하고, 
            # power_status와 operation_mode를 OFFLINE으로 명확히 기록
            sql = """
                INSERT INTO robot_status_log (
                    time, robot_id, power_status, operation_mode,
                    battery, is_charging, temperature, humidity, pm25, voc
                )
                VALUES (
                    CURRENT_TIMESTAMP, %s, 'OFFLINE', 'OFFLINE',
                    0, False, 0.0, 0.0, 0.0, 0
                );
            """
            cursor.execute(sql, (robot_id,))
            conn.commit()
            print(f"DB 저장 완료: [{robot_id}] 상태가 OFFLINE으로 기록되었습니다.")
        else:
            print("DB 연결 실패로 DB 저장을 건너뜁니다.")

        # ==========================================
        # 3. EC2 서버로 알림 전송 (프론트엔드 SSE 트리거용)
        # ==========================================
        if EC2_IP:
            # EC2_IP가 http:// 를 포함하지 않을 경우를 대비한 방어 로직
            base_url = EC2_IP if EC2_IP.startswith('http') else f"http://{EC2_IP}"
            ec2_url = f"{base_url}/api/alert"
            
            # EC2로 전송할 알림 페이로드 구성
            alert_payload = {
                "robot_id": robot_id,
                "alert_type": "LWT_OFFLINE",
                "status": status,
                "reason": reason,
                "message": f"로봇({robot_id})과 통신이 비정상적으로 두절되었습니다."
            }
            
            encoded_data = json.dumps(alert_payload).encode('utf-8')
            
            print(f"Target URL: {ec2_url}")
            response = http.request(
                'POST',
                ec2_url,
                body=encoded_data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"EC2 알림 전송 결과: {response.status}, {response.data.decode('utf-8')}")
        else:
            print("EC2_IP 환경변수가 설정되지 않아 알림 전송을 건너뜁니다.")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'message': 'LWT processing complete',
                'robot_id': robot_id
            })
        }

    except Exception as e:
        print(f"람다 실행 중 치명적 에러 발생: {str(e)}")
        if conn:
            conn.rollback()
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()