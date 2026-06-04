import pytest
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
