@echo off
:: 1. 경로 설정 및 로봇 ID 설정
:: 환경변수 설정을 명확히 하여 파이썬 코드에서도 인식하게 합니다.
set ROBOT_ID=robot_id=1
cd /d "C:\aria\ai\event_ai"

echo [MLOps] 1. %ROBOT_ID% 데이터 동기화 시작...
:: [수정] s3 최상위가 아니라 learning_data 폴더에서 가져오도록 변경
aws s3 cp s3://aria-learningdata-storage/%ROBOT_ID%/learning_data/ ./sync/data_lake --recursive --exclude "*" --include "*.json"

echo [MLOps] 2. 데이터 무결성 검사 및 리스트 갱신...
:: data_cleaner.py 실행 (sync 폴더 내 위치 확인)
python ./sync/data_cleaner.py

echo [MLOps] 3. 고도화 재학습 시작 (S3 가중치 자동 체크 포함)...
:: [수정] retrain.py가 training 폴더 안에 있으므로 경로 유지
python ./training/retrain.py

echo [MLOps] 4. S3 배포 및 이력 저장 시작...
:: [수정] deploy_model.py가 utils 폴더 안에 있으므로 경로 유지
python ./utils/deploy_model.py

echo 모든 과정이 완료되었습니다!
pause