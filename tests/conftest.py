import os
import csv
import sys
import time
import types
import pytest

if 'ultralytics' not in sys.modules:
    stub = types.ModuleType('ultralytics')
    class _FakeYOLO:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return []
        def predict(self, *a, **k): return []
    stub.YOLO = _FakeYOLO
    sys.modules['ultralytics'] = stub

if 'langchain_community' not in sys.modules:
    lc = types.ModuleType('langchain_community')
    lc_vs = types.ModuleType('langchain_community.vectorstores')
    lc_emb = types.ModuleType('langchain_community.embeddings')
    class _FakeFAISS:
        @staticmethod
        def load_local(*a, **k): return None
    class _FakeEmbeddings:
        def __init__(self, *a, **k): pass
    lc_vs.FAISS = _FakeFAISS
    lc_emb.HuggingFaceEmbeddings = _FakeEmbeddings
    sys.modules['langchain_community'] = lc
    sys.modules['langchain_community.vectorstores'] = lc_vs
    sys.modules['langchain_community.embeddings'] = lc_emb

from app import create_app
from app.models import db as _db

# 프로젝트 루트 기준 절대경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATA_PATH = os.path.join(DATA_DIR, 'processed_faers.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'ml')

@pytest.fixture(scope='session', autouse=True)
def ensure_sample_data():
    created = False
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        rows = [
            ['primaryid', 'drugname', 'pt', 'age', 'age_group', 'sex', 'outc_cod', 'reporter_country', 'sym_2024', 'quarter'],
            [1, 'METHOTREXATE', 'FATIGUE', 60, '60-69', 'M', 'OT', 'US', 1, '2024Q1'],
            [2, 'METHOTREXATE', 'FATIGUE', 62, '60-69', 'M', 'OT', 'US', 1, '2024Q1'],
            [3, 'ASPIRIN', 'BLEEDING', 70, '70-79', 'M', 'HO', 'KR', 1, '2024Q1'],
            [4, 'ASPIRIN', 'BLEEDING', 71, '70-79', 'M', 'HO', 'KR', 1, '2024Q1'],
            [5, 'WARFARIN', 'BLEEDING', 72, '70-79', 'M', 'HO', 'US', 1, '2024Q1'],
            [6, 'METHOTREXATE', 'NAUSEA', 45, '40-49', 'F', 'HO', 'KR', 1, '2024Q2'],
            [7, 'METHOTREXATE', 'NAUSEA', 50, '40-49', 'F', 'DE', 'US', 1, '2024Q2'],
            [8, 'METHOTREXATE', 'NAUSEA', 55, '50-59', 'F', 'OT', 'US', 1, '2024Q2'],
            [9, 'ASPIRIN', 'BLEEDING', 73, '70-79', 'M', 'HO', 'KR', 1, '2024Q2'],
            [10, 'ASPIRIN', 'NAUSEA', 65, '60-69', 'F', 'OT', 'KR', 1, '2024Q2'],
            [11, 'WARFARIN', 'BLEEDING', 74, '70-79', 'M', 'HO', 'US', 1, '2024Q2'],
        ]
        with open(DATA_PATH, 'w', newline='', encoding='cp949') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        created = True
    yield
    if created:
        os.remove(DATA_PATH)

@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        # CI 환경에서 경로가 누락되지 않도록 명시적으로 설정
        'DATA_PATH': DATA_PATH,
        'MODEL_DIR': MODEL_DIR,
    })
    with app.app_context():
        _db.create_all()
        yield app
        # SHAP/XGBoost C 확장 백그라운드 스레드 정리 대기 (Windows access violation 방지)
        time.sleep(0.5)
        _db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()
