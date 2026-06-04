from flask import Flask
from config import Config
from app.models import db
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_mail import Mail
from flask_restx import Api

cache = Cache()
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
login_manager = LoginManager()
mail = Mail()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    cache.init_app(app)
    limiter.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = '로그인이 필요해요!'
    mail.init_app(app)

    # 배포용 routes_lite 사용
    from app.routes_lite import main
    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app

@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))
