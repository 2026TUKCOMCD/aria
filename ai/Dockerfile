# 1. 파이썬 베이스 이미지 (라즈베리파이 64bit 권장)
FROM python:3.10.11-slim-bullseye

# 2. 시스템 의존성 설치 (OpenCV 및 PySerial용)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 요구사항 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 파이썬 경로 설정: 루트(/app)와 하위 모듈 폴더들을 모두 추가
ENV PYTHONPATH=/app:/app/event_ai:/app/event_ai/core

# 7. 실행
CMD ["python3", "event_ai/core/main.py"]