import os
import csv
import sys
import types
import pytest

# ultralytics(YOLO)는 무겁고 vision 기능 테스트와 무관하므로 스텁으로 대체
if 'ultralytics' not in sys.modules:
    stub = types.ModuleType('ultralytics')
    class _FakeYOLO:
        def __init__(self, *a, **k):
            pass
        def __call__(self, *a, **k):
            return []
        def predict(self, *a, **k):
            return []
    stub.YOLO = _FakeYOLO
    sys.modules['ultralytics'] = stub

if 'langchain_community' not in sys.modules:
    lc = types.ModuleType('langchain_community')
    lc_vs = types.ModuleType('langchain_community.vectorstores')
    lc_emb = types.ModuleType('langchain_community.embeddings')
    class _FakeFAISS:
        @staticmethod
        def load_local(*a, **k):
            return None
    class _FakeEmbeddings:
        def __init__(self, *a, **k):
            pass
    lc_vs.FAISS = _FakeFAISS
    lc_emb.HuggingFaceEmbeddings = _FakeEmbeddings
    sys.modules['langchain_community'] = lc
    sys.modules['langchain_community.vectorstores'] = lc_vs
    sys.modules['langchain_community.embeddings'] = lc_emb

from app import create_app
from app.models import db as _db

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
DATA_PATH = os.path.join(DATA_DIR, 'processed_faers.csv')


@pytest.fixture(scope='session', autouse=True)
def ensure_sample_data():
    """
    CI 등 실제 FAERS 데이터(480k행, gitignore)가 없는 환경에서
    /api/search, /api/autocomplete 등이 동작하도록 작은 샘플 CSV를 생성.
    실제 데이터 파일이 있으면 건드리지 않음.
    """
    created = False
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        rows = [
            ['primaryid', 'drugname', 'pt', 'age', 'age_group', 'sex', 'outc_cod', 'reporter_country', 'sym_2024'],
            [1, 'METHOTREXATE', 'NAUSEA', 45, '40-49', 'F', 'HO', 'KR', 1],
            [2, 'METHOTREXATE', 'FATIGUE', 60, '60-69', 'M', 'OT', 'US', 1],
            [3, 'METHOTREXATE', 'NAUSEA', 55, '50-59', 'F', 'DE', 'US', 1],
            [4, 'ASPIRIN', 'BLEEDING', 70, '70-79', 'M', 'HO', 'KR', 1],
            [5, 'ASPIRIN', 'NAUSEA', 65, '60-69', 'F', 'OT', 'KR', 1],
            [6, 'WARFARIN', 'BLEEDING', 72, '70-79', 'M', 'HO', 'US', 1],
        ]
        with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        created = True

    yield

    if created:
        os.remove(DATA_PATH)


@pytest.fixture
def app():
    """테스트용 Flask 앱 생성"""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',  # 메모리 DB 사용 (테스트 후 사라짐)
        'WTF_CSRF_ENABLED': False,
    })

    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()


@pytest.fixture
def client(app):
    """테스트 클라이언트"""
    return app.test_client()
