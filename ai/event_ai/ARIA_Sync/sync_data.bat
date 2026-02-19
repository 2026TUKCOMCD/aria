@echo off
:: 로봇 ID 설정 (S3 폴더명과 일치시켜야 함)
set ROBOT_ID=robot_id=1
cd /d "C:\aria\ai\event_ai"

echo [MLOps] 1. %ROBOT_ID% 데이터 동기화 시작...
:: --recursive를 사용해 하위 날짜 폴더 내의 모든 json을 한 폴더로 모읍니다.
aws s3 cp s3://aria-learningdata-storage/%ROBOT_ID%/ ./ARIA_Sync/data_lake --recursive --exclude "*" --include "*.json"

echo [MLOps] 2. 데이터 무결성 검사 및 리스트 갱신...
python ./ARIA_Sync/data_cleaner.py

echo [MLOps] 3. 고도화 재학습 시작...
python retrain.py

echo [MLOps] 4. S3 배포 폴더 업로드...
:: 배포는 날짜 폴더가 아닌 robot_id 바로 아래 deploy 폴더에 넣는 것이 관리하기 편합니다.
python deploy_model.py

echo 모든 과정이 완료되었습니다!
pause