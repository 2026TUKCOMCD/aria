# 3.10.11 버전으로 고정
FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir paho-mqtt psycopg2

COPY . .
CMD ["python", "app.py"]