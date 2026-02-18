@echo off
:: 파일이 위치한 폴더로 이동
cd /d "C:\aria\ai\event_ai\ARIA_Sync"

echo [ARIA MLOps] 1. S3 데이터 동기화 시작...
aws s3 sync s3://aria-learningdata-storage ./data_lake --delete

echo [ARIA MLOps] 2. 데이터 무결성 검사 및 목록 갱신...
python data_cleaner.py

:: 3. 재학습 단계 추가
echo [ARIA MLOps] 3. 신규 데이터 기반 GPU 재학습 시작...
:: 한 단계 위 폴더에 있는 retrain.py를 실행
python ../retrain.py

echo.
echo [완료] 동기화부터 재학습까지 모든 공정이 끝났습니다.
echo 생성된 모델: C:\aria\ai\event_ai\refined_model.pt
pause