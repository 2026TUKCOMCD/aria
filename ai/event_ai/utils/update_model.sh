#!/bin/bash

# 1. S3 디렉토리 구조 일치 (robot_id=1 형식)
ROBOT_ID="robot_id=1"
BUCKET="aria-learningdata-storage"

# 2. 절대 경로 설정
PROJECT_DIR="/srv/aria/users/sk/ai/event_ai"
MODEL_NAME="event_model.pt"
TEMP_MODEL="event_model_new.pt"

echo "[$ROBOT_ID] 최신 지능 체크 중..."

# S3에서 최신 모델 다운로드 (상세 경로 반영)
aws s3 cp s3://${BUCKET}/${ROBOT_ID}/deploy/event_model_latest.pt ${PROJECT_DIR}/${TEMP_MODEL}

# 파일이 정상적으로 다운로드 되었는지 확인
if [ -f "${PROJECT_DIR}/${TEMP_MODEL}" ]; then
    # 기존 모델과 비교
    if ! cmp -s "${PROJECT_DIR}/${MODEL_NAME}" "${PROJECT_DIR}/${TEMP_MODEL}"; then
        echo "새 모델이 발견되었습니다! 업데이트 중..."
        mv "${PROJECT_DIR}/${TEMP_MODEL}" "${PROJECT_DIR}/${MODEL_NAME}"
        echo "업데이트 완료: $(date)"
    else
        echo "이미 최신 모델입니다."
        rm "${PROJECT_DIR}/${TEMP_MODEL}"
    fi
else
    echo "S3에서 모델을 가져오지 못했습니다. 경로를 확인하세요."
fi