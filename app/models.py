from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    """사용자 계정"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M')
        }


class DrugSearch(db.Model):
    """약물 검색 기록"""
    __tablename__ = 'drug_searches'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    drugname = db.Column(db.String(100), nullable=False)
    total_reports = db.Column(db.Integer)
    age_avg = db.Column(db.Float)
    searched_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'drugname': self.drugname,
            'total_reports': self.total_reports,
            'age_avg': self.age_avg,
            'searched_at': self.searched_at.strftime('%Y-%m-%d %H:%M')
        }


class FavoriteDrug(db.Model):
    """즐겨찾기 약물"""
    __tablename__ = 'favorite_drugs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    drugname = db.Column(db.String(100), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'drugname': self.drugname,
            'added_at': self.added_at.strftime('%Y-%m-%d %H:%M')
        }


class PredictionLog(db.Model):
    """AI 예측 기록"""
    __tablename__ = 'prediction_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    drugname = db.Column(db.String(100), nullable=False)
    reaction = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Float)
    sex = db.Column(db.String(1))
    risk = db.Column(db.Integer)
    safe_prob = db.Column(db.Float)
    risk_prob = db.Column(db.Float)
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'drugname': self.drugname,
            'reaction': self.reaction,
            'age': self.age,
            'sex': self.sex,
            'risk': self.risk,
            'safe_prob': self.safe_prob,
            'risk_prob': self.risk_prob,
            'predicted_at': self.predicted_at.strftime('%Y-%m-%d %H:%M')
        }