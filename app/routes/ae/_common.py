"""
app/routes/ae/_common.py
AE 모듈 공통 함수 및 상수
- 감사로그 기록, CTCAE 자동 분류, PDF 테이블 생성
"""


from flask import request
from flask_login import current_user
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import cm
from app.models import db, AuditTrail

# CTCAE 자동 분류 키워드
CTCAE_KEYWORDS = {
    5: ['death', 'fatal', '사망'],
    4: ['life-threatening', 'life threatening', '생명위협', 'icu', 'ventilat'],
    3: ['hospitali', 'severe', '입원', '중증', 'severe'],
    2: ['moderate', '중증도', 'limiting'],
    1: ['mild', '경미', 'minor'],
}
SAE_KEYWORDS = ['사망', '입원', '생명위협', '영구장애', '선천성이상', 'death', 'hospitali',
                'life-threatening', 'disability', 'congenital']

def log_audit(action, table_name, record_id=None, old_value=None, new_value=None, reason=None):
    """21 CFR Part 11 Audit Trail 자동 기록"""
    try:
        username = current_user.username if current_user.is_authenticated else 'anonymous'
        user_id = current_user.id if current_user.is_authenticated else None
        trail = AuditTrail(
            user_id=user_id,
            username=username,
            action=action,
            table_name=table_name,
            record_id=record_id,
            old_value=str(old_value) if old_value else None,
            new_value=str(new_value) if new_value else None,
            ip_address=request.remote_addr,
            reason=reason
        )
        db.session.add(trail)
        db.session.commit()
    except:
        pass    # 감사 기록 실패해도 본 작업에는 영향 없도록 설정.

def auto_ctcae_grade(ae_term: str) -> int:
    term_lower = ae_term.lower()
    for grade in [5,4,3,2,1]:
        for kw in CTCAE_KEYWORDS[grade]:
            if kw in term_lower:
                return grade
    return 1


def auto_is_sae(ae_term: str, ctcae_grade: int) -> bool:
    if ctcae_grade >= 3:
        return True
    term_lower = ae_term.lower()
    return any(kw in term_lower for kw in SAE_KEYWORDS)

def _make_table(data):
    t = Table(data, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a56db')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    return t

CONSENT_VERSION = "v1.0"

VALID_MEANINGS = ["승인", "검토", "책임자 확인"]

CONSENT_TEXT = (
    "본 전자서명은 익명화 된 FDA FAERS 공개 데이터를 기반으로 분석된 "
    "이상사례(AE) 평가 결과에 대한 검토와 승인 행위를 기록하기 위한 것입니다. "
    "본 시스템은 실제 환자 식별정보를 수집, 저장하지 않으며, "
    "공식 규제기관(식약처, FDA 등) 제출용 법적 문서가 아닌 "
    "포트폴리오용 목적으로 만들어진 약물감시(Pharmacovigilance) 분석 도구입니다. "
    "서명자는 본인의 검토 행위가 위 사실에 기반함을 이해하고 동의합니다."
)
