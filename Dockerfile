FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless \
    && pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

EXPOSE 5001

CMD gunicorn run:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
