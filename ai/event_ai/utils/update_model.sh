#!/bin/bash

# 1. 환경 설정
ROBOT_ID="robot_id=1"
BUCKET="aria-learningdata-storage"
# 라즈베리파이 내 실제 프로젝트 경로 (성국 님의 환경에 맞춰 확인 필요)
PROJECT_DIR="/srv/aria/users/sk/ai/event_ai"

# 파일명 정의
MODEL_NAME="event_model.pt"
SCALER_NAME="scaler.pkl"
S3_MODEL_LATEST="event_model_latest.pt"

# S3 경로 설정
S3_CURRENT_PATH="s3://${BUCKET}/${ROBOT_ID}/weight/current"

echo "----------------------------------------------------"
echo "[$(date)] [$ROBOT_ID] 최신 지능 체크 시작"
echo "----------------------------------------------------"

# 2. S3 'current' 디렉토리에서 최신 모델과 스케일러 다운로드 시도
# 임시 파일명으로 다운로드하여 기존 파일과 비교 준비
aws s3 cp ${S3_CURRENT_PATH}/${S3_MODEL_LATEST} ${PROJECT_DIR}/temp_${MODEL_NAME}
aws s3 cp ${S3_CURRENT_PATH}/${SCALER_NAME} ${PROJECT_DIR}/temp_${SCALER_NAME}

# 3. 모델 파일 다운로드 성공 여부 확인
if [ -f "${PROJECT_DIR}/temp_${MODEL_NAME}" ]; then
    
    # 4. 기존 모델과 새로운 모델 비교 (변경 사항이 있는지 확인)
    if ! cmp -s "${PROJECT_DIR}/${MODEL_NAME}" "${PROJECT_DIR}/temp_${MODEL_NAME}"; then
        echo ">> 새로운 모델이 발견되었습니다! 업데이트를 진행합니다."
        
        # 모델 교체
        mv "${PROJECT_DIR}/temp_${MODEL_NAME}" "${PROJECT_DIR}/${MODEL_NAME}"
        
        # 스케일러 교체 (세트로 업데이트)
        if [ -f "${PROJECT_DIR}/temp_${SCALER_NAME}" ]; then
            mv "${PROJECT_DIR}/temp_${SCALER_NAME}" "${PROJECT_DIR}/${SCALER_NAME}"
            echo ">> 스케일러 업데이트 완료"
        fi
        
        echo ">> 업데이트 성공: $(date)"
    else
        echo ">> 현재 이미 최신 모델을 사용 중입니다."
        # 변경 사항 없으므로 임시 파일 삭제
        rm "${PROJECT_DIR}/temp_${MODEL_NAME}"
        rm -f "${PROJECT_DIR}/temp_${SCALER_NAME}"
    fi
else
    echo ">> [경고] S3에서 모델을 가져오지 못했습니다."
    echo ">> S3 경로(${S3_CURRENT_PATH}) 또는 권한을 확인하세요."
fi

echo "----------------------------------------------------"
echo "[$(date)] 체크 종료"
echo "----------------------------------------------------"