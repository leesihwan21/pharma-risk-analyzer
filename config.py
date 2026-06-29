import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'pharma-risk-dev-key'

    # DB
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///pharma.db'
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 캐시
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300

    # 메일
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') or 'your-app-password'
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'

    # API 키
    MFDS_API_KEY = os.environ.get('MFDS_API_KEY') or ''
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY') or ''

    # ── 경로 (하드코딩 제거) ──────────────────────────
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'ml')
    RAG_DB_PATH = os.path.join(BASE_DIR, 'rag_db')
    FONT_PATH = os.path.join(BASE_DIR, 'NanumGothic.ttf')
    PILL_DB_PATH = os.path.join(BASE_DIR, 'data', 'pill_identity.db')
    MLFLOW_DB_PATH = os.path.join(BASE_DIR, 'mlflow.db')
    PIPELINE_LOG_PATH = os.path.join(BASE_DIR, 'ml', 'pipeline.log.json')

    # ── 모델명 (하드코딩 제거) ────────────────────────
    CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-6')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    # ── 상수 (하드코딩 제거) ─────────────────────────
    RAG_TOP_K = 5
    PUBMED_MAX_RESULTS = 5
    API_TIMEOUT = 60
    AI_TIMEOUT = 120