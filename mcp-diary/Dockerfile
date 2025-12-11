# Python 3.11 slim 이미지 기반
FROM python:3.11-slim

# 시스템 패키지 업데이트 및 기본 유틸 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# Python 의존성 사전 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY diary_mcp.py .

# FastAPI 서버 실행 (포트 7002)
EXPOSE 7002

# uvicorn 실행
CMD ["uvicorn", "diary_mcp:app", "--host", "0.0.0.0", "--port", "7002"]
