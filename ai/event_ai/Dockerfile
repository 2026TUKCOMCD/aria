FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# 2. 필수 라이브러리 설치 (기존 방식 유지 및 확장)

# psycopg2-binary, numpy, pandas, scikit-learn 포함
RUN pip install --no-cache-dir pandas scikit-learn psycopg2-binary numpy
WORKDIR /workspace

# 3. 소스 코드 복사
COPY . .

# 4. 환경 검증을 위해 첫 실행 시 check_env.py를 실행
# 나중에 실전용으로 바꿀 때 CMD ["python", "inference.py"]로 수정
CMD ["python", "check_env.py"]
CMD ["tail", "-f", "/dev/null"]

