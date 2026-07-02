"""
app/routes/ae/core.py
AE(이상사례) CRUD - 등록/조회/수정/삭제/제출
"""
from datetime import datetime, timedelta, UTC
from flask import render_template, jsonify, request
from flask_login import login_required

from . import ae
from ._common import log_audit, auto_ctcae_grade, auto_is_sae
from app.models import db, AEReport


@ae.route('/ae_manager')
def ae_manager():
    return render_template('ae_manager.html')


@ae.route('/ae/list')
@login_required
def ae_list():
    reports = AEReport.query.order_by(AEReport.reported_at.desc()).all()
    return render_template('ae_list.html', reports=reports)


@ae.route('/api/ae/list')
def ae_list_api():
    status_filter = request.args.get('status', '')
    sae_only = request.args.get('sae_only', 'false') == 'true'

    query = AEReport.query.order_by(AEReport.reported_at.desc())
    if sae_only:
        query = query.filter_by(is_sae=True)

    reports = query.all()
    result = [r.to_dict() for r in reports]
    if status_filter:
        result = [r for r in result if r['deadline_status'] == status_filter]

    all_reports = [r.to_dict() for r in AEReport.query.all()]
    summary = {
        'total': len(all_reports),
        'sae_count': sum(1 for r in all_reports if r['is_sae']),
        'overdue': sum(1 for r in all_reports if r['deadline_status'] == 'overdue'),
        'urgent': sum(1 for r in all_reports if r['deadline_status'] == 'urgent'),
        'submitted': sum(1 for r in all_reports if r['is_submitted']),
    }
    return jsonify({'reports': result, 'summary': summary})


@ae.route('/api/ae/<int:ae_id>')
def ae_detail(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    return jsonify(report.to_dict())


@ae.route('/api/ae/create', methods=['POST'])
def ae_create():
    data = request.get_json()

    if not data.get('patient_code') or not data.get('drugname') or not data.get('ae_term'):
        return jsonify({'error': '환자코드, 약물명, AE 용어는 필수입니다'}), 400

    ae_term = data.get('ae_term', '')
    ctcae_grade = int(data.get('ctcae_grade') or auto_ctcae_grade(ae_term))

    is_sae_input = data.get('is_sae')
    is_sae = bool(is_sae_input) if is_sae_input is not None else auto_is_sae(ae_term, ctcae_grade)

    report_deadline = datetime.now(UTC) + timedelta(days=15) if is_sae else None

    ae_start = None
    ae_end = None
    try:
        if data.get('ae_start_date'):
            ae_start = datetime.strptime(data['ae_start_date'], '%Y-%m-%d').date()
        if data.get('ae_end_date'):
            ae_end = datetime.strptime(data['ae_end_date'], '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': '날짜 형식이 올바르지 않습니다 (YYYY-MM-DD)'}), 400

    report = AEReport(
        patient_code=data.get('patient_code', '').upper(),
        age=float(data['age']) if data.get('age') else None,
        sex=data.get('sex', ''),
        drugname=data.get('drugname', '').upper(),
        dose=data.get('dose', ''),
        route=data.get('route', ''),
        ae_term=ae_term,
        ae_start_date=ae_start,
        ae_end_date=ae_end,
        ctcae_grade=ctcae_grade,
        is_sae=is_sae,
        sae_category=data.get('sae_category', ''),
        causality=data.get('causality', ''),
        action_taken=data.get('action_taken', ''),
        outcome=data.get('outcome', ''),
        report_deadline=report_deadline,
        is_submitted=False,
        notes=data.get('notes', ''),
    )

    try:
        db.session.add(report)
        db.session.commit()
        log_audit('CREATE', 'ae_reports', record_id=report.id, new_value=report.to_dict())
        return jsonify({
            'message': 'AE 보고서가 등록됐습니다',
            'id': report.id,
            'is_sae': is_sae,
            'ctcae_grade': ctcae_grade,
            'report_deadline': report_deadline.strftime('%Y-%m-%d') if report_deadline else None
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@ae.route('/api/ae/<int:ae_id>/update', methods=['POST'])
def ae_update(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    data = request.get_json()

    fields = ['ae_term', 'ctcae_grade', 'is_sae', 'sae_category',
              'causality', 'action_taken', 'outcome', 'notes',
              'dose', 'route', 'age', 'sex']
    for f in fields:
        if f in data:
            setattr(report, f, data[f])

    if 'is_sae' in data:
        if data['is_sae'] and not report.report_deadline:
            report.report_deadline = datetime.now(UTC) + timedelta(days=15)
        elif not data['is_sae']:
            report.report_deadline = None

    try:
        db.session.commit()
        log_audit('UPDATE', 'ae_reports', record_id=ae_id, reason=data.get('reason', ''))
        return jsonify({'message': '수정됐습니다', 'report': report.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@ae.route('/api/ae/<int:ae_id>/submit', methods=['POST'])
def ae_submit(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    report.is_submitted = True
    try:
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} 규제기관 제출 완료'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@ae.route('/api/ae/<int:ae_id>/delete', methods=['POST'])
def ae_delete(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    try:
        db.session.delete(report)
        log_audit('DELETE', 'ae_reports', record_id=ae_id, old_value=report.to_dict())
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} 삭제됐습니다'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500