from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timedelta

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


# ──────────────────────────────────────────
# AE (Adverse Event) 관리 모듈
# ──────────────────────────────────────────

class AEReport(db.Model):
    """이상사례(AE) 보고서"""
    __tablename__ = 'ae_reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # 환자 정보
    patient_code = db.Column(db.String(50), nullable=False)   # 환자 식별코드 (익명)
    age = db.Column(db.Float, nullable=True)
    sex = db.Column(db.String(1), nullable=True)               # M / F

    # 약물 정보
    drugname = db.Column(db.String(200), nullable=False)       # 시험약명
    dose = db.Column(db.String(100), nullable=True)            # 용량
    route = db.Column(db.String(50), nullable=True)            # 투여경로 (경구/정맥 등)

    # 이상사례 정보
    ae_term = db.Column(db.String(200), nullable=False)        # AE 용어 (MedDRA PT 기준)
    ae_start_date = db.Column(db.Date, nullable=True)          # AE 발생일
    ae_end_date = db.Column(db.Date, nullable=True)            # AE 종료일 (미종료면 null)

    # 중증도 분류 (CTCAE Grade 1~5)
    ctcae_grade = db.Column(db.Integer, nullable=True)         # 1~5
    # Grade 1: 경미 / 2: 중등도 / 3: 중증 / 4: 생명위협 / 5: 사망

    # SAE 여부 및 분류
    is_sae = db.Column(db.Boolean, default=False)              # SAE 해당 여부
    sae_category = db.Column(db.String(100), nullable=True)
    # 사망 / 입원 / 생명위협 / 영구장애 / 선천성이상 / 기타 중요 의학적 사건

    # 인과관계 평가
    causality = db.Column(db.String(50), nullable=True)
    # Certain / Probable / Possible / Unlikely / Unclassifiable

    # 처리 및 결과
    action_taken = db.Column(db.String(100), nullable=True)    # 투여중단/감량/유지/해당없음
    outcome = db.Column(db.String(100), nullable=True)         # 회복/회복중/미회복/사망/불명

    # 보고 관련
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)   # 보고 입력일
    report_deadline = db.Column(db.DateTime, nullable=True)          # 보고 마감일
    # SAE: 입력일로부터 15일 이내 규제기관 보고 의무
    is_submitted = db.Column(db.Boolean, default=False)              # 규제기관 제출 여부
    notes = db.Column(db.Text, nullable=True)                        # 비고

    def days_until_deadline(self):
        """보고 마감까지 남은 일수"""
        if self.report_deadline and not self.is_submitted:
            delta = self.report_deadline - datetime.utcnow()
            return delta.days
        return None

    def deadline_status(self):
        """마감 상태 반환"""
        days = self.days_until_deadline()
        if days is None:
            return 'submitted' if self.is_submitted else 'no_deadline'
        if days < 0:
            return 'overdue'       # 기한 초과
        if days <= 3:
            return 'urgent'        # 3일 이내 긴급
        if days <= 7:
            return 'warning'       # 7일 이내 주의
        return 'normal'

    def to_dict(self):
        return {
            'id': self.id,
            'patient_code': self.patient_code,
            'age': self.age,
            'sex': self.sex,
            'drugname': self.drugname,
            'dose': self.dose,
            'route': self.route,
            'ae_term': self.ae_term,
            'ae_start_date': self.ae_start_date.strftime('%Y-%m-%d') if self.ae_start_date else None,
            'ae_end_date': self.ae_end_date.strftime('%Y-%m-%d') if self.ae_end_date else None,
            'ctcae_grade': self.ctcae_grade,
            'is_sae': self.is_sae,
            'sae_category': self.sae_category,
            'causality': self.causality,
            'action_taken': self.action_taken,
            'outcome': self.outcome,
            'reported_at': self.reported_at.strftime('%Y-%m-%d %H:%M'),
            'report_deadline': self.report_deadline.strftime('%Y-%m-%d') if self.report_deadline else None,
            'days_until_deadline': self.days_until_deadline(),
            'deadline_status': self.deadline_status(),
            'is_submitted': self.is_submitted,
            'notes': self.notes,
        }
