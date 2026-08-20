import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from pathlib import Path

# 1. .env 파일에서 환경 변수(ROBOT_ID) 로드
dotenv_path = Path("/srv/aria/users/hs/aria/.env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# 로봇 ID 가져오기 (기본값: '1')
robot_id = os.environ.get('ROBOT_ID', '1')

# 2. S3 버킷 정보 설정
S3_BUCKET_NAME = 'aria-learningdata-storage'

# S3 내부의 경로를 'robot_id=1/data/aria_daily_log.csv' 형태로 동적 생성
S3_OBJECT_NAME = f'robot_id={robot_id}/data/aria_daily_log.csv'

# 라즈베리파이 로컬 파일 경로
LOCAL_DATA_PATH = os.path.expanduser('~/aria_ai_system/data/aria_daily_log.csv')

def get_s3_client():
    """Boto3 S3 클라이언트를 생성하여 반환합니다."""
    try:
        s3 = boto3.client('s3')
        return s3
    except NoCredentialsError:
        print("❌ AWS 자격 증명(Credentials)을 찾을 수 없습니다. aws configure를 확인하세요.")
        return None

def upload_to_s3():
    """낮 동안 모은 로컬 CSV 파일을 S3 버킷으로 업로드합니다."""
    s3 = get_s3_client()
    if not s3:
        return False

    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"⚠️ 업로드할 파일이 없습니다: {LOCAL_DATA_PATH}")
        return False

    try:
        print(f"☁️ S3에 업로드 중... [{LOCAL_DATA_PATH} -> s3://{S3_BUCKET_NAME}/{S3_OBJECT_NAME}]")
        s3.upload_file(LOCAL_DATA_PATH, S3_BUCKET_NAME, S3_OBJECT_NAME)
        print("✅ S3 업로드 성공!")
        return True
    except ClientError as e:
        print(f"❌ S3 업로드 실패: {e}")
        return False

def download_from_s3(download_path=LOCAL_DATA_PATH):
    """심야 학습 전, S3에서 최신 CSV 파일을 다운로드합니다."""
    s3 = get_s3_client()
    if not s3:
        return False

    try:
        print(f"☁️ S3에서 다운로드 중... [s3://{S3_BUCKET_NAME}/{S3_OBJECT_NAME} -> {download_path}]")
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        s3.download_file(S3_BUCKET_NAME, S3_OBJECT_NAME, download_path)
        print("✅ S3 다운로드 성공!")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"⚠️ S3 버킷에 아직 다운로드할 로그 파일이 없습니다. (경로: {S3_OBJECT_NAME})")
        else:
            print(f"❌ S3 다운로드 실패: {e}")
        return False
