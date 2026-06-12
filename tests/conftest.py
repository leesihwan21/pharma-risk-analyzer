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
