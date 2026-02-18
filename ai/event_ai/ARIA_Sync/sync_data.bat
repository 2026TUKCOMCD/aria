@echo off
:: 파일이 위치한 폴더로 이동
cd /d "C:\aria\ai\event_ai\ARIA_Sync"

echo 🔄 [ARIA MLOps] S3 데이터 동기화 시작...
aws s3 sync s3://aria-learningdata-storage ./data_lake --delete

echo 🧹 [ARIA MLOps] 데이터 무결성 검사 실행...
python data_cleaner.py

echo.
echo ✅ 모든 작업이 완료되었습니다.
pause