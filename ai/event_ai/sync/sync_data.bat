@echo off
:: 1. 경로 설정 및 로봇 ID 설정
set ROBOT_ID=robot_id=1
cd /d "C:\aria\ai\event_ai"

echo [MLOps] 1. %ROBOT_ID% 데이터 동기화 시작...
:: 저장 위치를 새로 만든 sync/data_lake 폴더로 변경합니다.
aws s3 cp s3://aria-learningdata-storage/%ROBOT_ID%/ ./sync/data_lake --recursive --exclude "*" --include "*.json"

echo [MLOps] 2. 데이터 무결성 검사 및 리스트 갱신...
:: data_cleaner.py가 sync 폴더 안에 있으므로 경로 수정
python ./sync/data_cleaner.py

echo [MLOps] 3. 고도화 재학습 시작...
:: retrain.py(또는 train.py)가 training 폴더 안에 있으므로 경로 수정
python ./training/retrain.py

echo [MLOps] 4. S3 배포 폴더 업로드...
:: deploy_model.py가 utils 폴더 안에 있으므로 경로 수정
python ./utils/deploy_model.py

echo 모든 과정이 완료되었습니다!
pause