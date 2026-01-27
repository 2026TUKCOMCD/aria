FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 필요한 추가 라이브러리가 있다면 여기에 작성 (예: pandas, scikit-learn 등)
RUN pip install --no-cache-dir pandas scikit-learn psycopg2-binary

WORKDIR /workspace

# 컨테이너 실행 시 main.py 대기 (나중에 실제 모델 코드로 대체)
CMD ["python", "main.py"]