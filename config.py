import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'pharma-risk-dev-key'
    
    # SQLite → PostgreSQL 전환
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///pharma.db'
    # Railway PostgreSQL은 postgres:// 로 시작하는데 SQLAlchemy는 postgresql:// 필요
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') or 'your-app-password'
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'
    MFDS_API_KEY = os.environ.get('MFDS_API_KEY') or ''
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY') or ''