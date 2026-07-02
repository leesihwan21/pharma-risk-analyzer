"""
app/routes/ae/signature.py
21 CFR Part 11 전자서명 및 감사로그(Audit Trail)
"""
import hashlib
from datetime import datetime, UTC
from flask import jsonify, request
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash

from . import ae
from ._common import log_audit, CONSENT_VERSION, VALID_MEANINGS, CONSENT_TEXT
from app.models import db, AEReport, AuditTrail, ElectronicSignature


@ae.route('/api/audit-trail')
def get_audit_trail():
    logs = AuditTrail.query.order_by(AuditTrail.timestamp.desc()).limit(50).all()
    return jsonify({'logs': [l.to_dict() for l in logs]})


@ae.route('/api/ae/<int:ae_id>/sign', methods=['POST'])
@login_required
def sign_ae(ae_id):
    """21 CFR Part 11 전자서명"""
    report = AEReport.query.get_or_404(ae_id)
    data = request.get_json()
    password = data.get('password', '')
    meaning = data.get('meaning', '')
    reason = data.get('reason', '')
    consent_agreed = data.get('consent_agreed', False)

    if not password:
        return jsonify({'error': '비밀번호를 입력하세요'}), 400
    elif not reason:
        return jsonify({'error': '서명 사유를 입력하세요'}), 400
    elif meaning not in VALID_MEANINGS:
        return jsonify({'error': f'서명 의미는 {", ".join(VALID_MEANINGS)} 중 하나여야 합니다'}), 400
    elif not consent_agreed:
        log_audit('SIGN_FAILED', 'ae_reports', record_id=ae_id, reason='약관 미동의')
        return jsonify({'error': '약관에 동의해야 서명할 수 있습니다'}), 400

    elif not check_password_hash(current_user.password_hash, password):
        log_audit('SIGN_FAILED', 'ae_reports', record_id=ae_id, reason='비밀번호 불일치')
        return jsonify({'error': '비밀번호가 일치하지 않습니다'}), 401

    sign_data = f"{current_user.username}:{ae_id}:{meaning}:{datetime.now(UTC).isoformat()}"
    signature_hash = hashlib.sha256(sign_data.encode()).hexdigest()

    sig = ElectronicSignature(
        ae_report_id=ae_id,
        user_id=current_user.id,
        username=current_user.username,
        signer_role=current_user.role,
        signature_hash=signature_hash,
        meaning=meaning,
        reason=reason,
        consent_agreed=consent_agreed,
        consent_version=CONSENT_VERSION,
        ip_address=request.remote_addr
    )
    db.session.add(sig)

    report.is_submitted = True
    db.session.commit()

    log_audit('SIGN', 'ae_reports', record_id=ae_id,
              new_value=f'signed by {current_user.username} ({current_user.role})', reason=reason)

    return jsonify({
        'message': f'전자서명 완료: {meaning}',
        'signature_hash': signature_hash[:16] + '...',
        'signed_by': current_user.username,
        'signer_role': current_user.role,
        'signed_at': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    })


@ae.route('/api/consent-text')
def get_consent_text():
    """서명 전 표시할 약관 동의 문구"""
    return jsonify({
        'version': CONSENT_VERSION,
        'text': CONSENT_TEXT
    })


@ae.route('/api/ae/<int:ae_id>/signatures')
def get_signatures(ae_id):
    sigs = ElectronicSignature.query.filter_by(ae_report_id=ae_id).all()
    return jsonify({'signatures': [s.to_dict() for s in sigs]})