FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 1. 필수 라이브러리 설치
RUN pip install --no-cache-dir pandas scikit-learn psycopg2-binary numpy

# 작업 디렉토리 설정
WORKDIR /workspace

# 2. 소스 코드 복사 (전체 구조를 그대로 컨테이너로 복사)
COPY . .

# 3. 환경 변수 설정 (파이썬이 core, training 폴더를 인식하게 함)
ENV PYTHONPATH=/workspace

# 4. 실행 스크립트 경로 수정
CMD ["python", "utils/check_env.py"]

# 컨테이너 유지용 (개발 단계)
# CMD ["tail", "-f", "/dev/null"]