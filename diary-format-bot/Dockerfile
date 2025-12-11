FROM python:3.11-slim

WORKDIR /app

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir requests

# 소스 복사
COPY main.py /app/main.py

# 컨테이너 시작 시 실행
CMD ["python", "main.py"]
