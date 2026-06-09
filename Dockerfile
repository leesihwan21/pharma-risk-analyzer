FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD gunicorn run:app --bind 0.0.0.0:$PORT
