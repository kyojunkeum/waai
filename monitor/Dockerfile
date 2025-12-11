# 기존 python:3.11-slim 같은 이미지 대신 CUDA 런타임 사용
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Python 설치
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 필요한 패키지 설치 (psutil 포함)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 모니터 코드 복사
COPY monitor.py .

CMD ["python3", "monitor.py"]


