import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'pharma-risk-dev-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///pharma.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # 캐시 설정
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300

    # 메일 설정 (Gmail)
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') or 'your-app-password'
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'