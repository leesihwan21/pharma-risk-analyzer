from flask import Flask
from config import Config
from app.models import db
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api, Resource, fields
from flask_login import LoginManager
from flask_mail import Mail

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
    login_manager.login_view = 'auth.login'
    login_manager.login_message = '로그인이 필요합니다.'
    mail.init_app(app)

    from app.routes import main, auth, drug, ae, analysis, vision, literature
    from app.routes.rag import rag
    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(drug)
    app.register_blueprint(ae)
    app.register_blueprint(analysis)
    app.register_blueprint(vision)
    app.register_blueprint(literature)
    app.register_blueprint(rag)

    # Swagger API 문서
    api = Api(app,
        version='1.0',
        title='Pharma Risk Analyzer API',
        description='FDA FAERS 기반 약물 이상반응 분석 REST API',
        doc='/api/docs',
        prefix='/api/v1'
    )

    # 네임스페이스
    ns_drug = api.namespace('drugs', description='약물 검색 API')
    ns_predict = api.namespace('predict', description='AI 예측 API')

    # 모델 정의
    drug_model = api.model('DrugSearch', {
        'drug': fields.String(description='Drug Name'),
        'total_reports': fields.Integer(description='Total Reports'),
        'age_avg': fields.Float(description='Average Age'),
    })

    predict_input = api.model('PredictInput', {
        'drugname': fields.String(required=True, description='Drug Name', example='METHOTREXATE'),
        'reaction': fields.String(required=True, description='Adverse Reaction', example='FATIGUE'),
        'age': fields.Float(description='Age', example=50),
        'sex': fields.String(description='Sex (M/F)', example='F'),
    })

    predict_output = api.model('PredictOutput', {
        'drug': fields.String(description='Drug Name'),
        'reaction': fields.String(description='Adverse Reaction'),
        'risk': fields.Integer(description='Risk (0/1)'),
        'risk_label': fields.String(description='Risk Label'),
        'probability': fields.Raw(description='Probability'),
    })

    @ns_drug.route('/search/<string:drugname>')
    class DrugSearchAPI(Resource):
        @ns_drug.doc('drug_search')
        @ns_drug.marshal_with(drug_model)
        def get(self, drugname):
            """Search drug by name"""
            from app.routes.drug import get_drug_summary
            summary = get_drug_summary(drugname)
            if summary is None:
                api.abort(404, f'Drug not found: {drugname}')
            return summary

    @ns_predict.route('/risk')
    class PredictAPI(Resource):
        @ns_predict.doc('predict_risk')
        @ns_predict.expect(predict_input)
        @ns_predict.marshal_with(predict_output)
        def post(self):
            """AI-based drug adverse event risk prediction"""
            from app.routes.drug import predict_risk
            data = api.payload
            drugname = data.get('drugname', '')
            reaction = data.get('reaction', '')
            age = float(data.get('age', 50))
            sex = data.get('sex', 'F')

            try:
                return predict_risk(drugname, reaction, age, sex)
            except ValueError as e:
                api.abort(400, str(e))

    with app.app_context():
        db.create_all()

    return app

@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))
