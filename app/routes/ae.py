import io
import os
import math

from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, request, send_file
from flask_login import login_required, current_user
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from app.models import db, AEReport, AuditTrail

ae = Blueprint('ae', __name__)

# CTCAE 자동 분류 키워드
CTCAE_KEYWORDS = {
    5: ['death', 'fatal', '사망'],
    4: ['life-threatening', 'life threatening', '생명위협', 'icu', 'ventilat'],
    3: ['hospitali', 'severe', '입원', '중증', 'severe'],
    2: ['moderate', '중등도', 'limiting'],
    1: ['mild', '경미', 'minor'],
}

SAE_KEYWORDS = ['사망', '입원', '생명위협', '영구장애', '선천성이상', 'death', 'hospitali',
                'life-threatening', 'disability', 'congenital']

def log_audit(action, table_name, record_id=None, old_value=None, new_value=None, reason=None):
    """21 CFR Part 11 Audit Trail 자동 기록"""
    try:
        from flask import request as req
        from flask_login import current_user
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
            ip_address=req.remote_addr,
            reason=reason
        )
        db.session.add(trail)
        db.session.commit()
    except:
        pass  # 감사 기록 실패해도 본 작업에는 영향 없도록

def auto_ctcae_grade(ae_term: str) -> int:
    term_lower = ae_term.lower()
    for grade in [5, 4, 3, 2, 1]:
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

    report_deadline = datetime.utcnow() + timedelta(days=15) if is_sae else None

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
            report.report_deadline = datetime.utcnow() + timedelta(days=15)
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

@ae.route('/api/ae/stats')
def ae_stats():
    reports = AEReport.query.all()
    if not reports:
        return jsonify({'message': '등록된 AE가 없습니다'})

    grade_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    causality_dist = {}
    outcome_dist = {}

    for r in reports:
        if r.ctcae_grade:
            grade_dist[r.ctcae_grade] = grade_dist.get(r.ctcae_grade, 0) + 1
        if r.causality:
            causality_dist[r.causality] = causality_dist.get(r.causality, 0) + 1
        if r.outcome:
            outcome_dist[r.outcome] = outcome_dist.get(r.outcome, 0) + 1

    return jsonify({
        'total': len(reports),
        'sae_count': sum(1 for r in reports if r.is_sae),
        'submitted_count': sum(1 for r in reports if r.is_submitted),
        'grade_distribution': grade_dist,
        'causality_distribution': causality_dist,
        'outcome_distribution': outcome_dist,
    })

@ae.route('/api/ae/<int:ae_id>/pdf')
def ae_pdf(ae_id):
    report = AEReport.query.get_or_404(ae_id)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    title_style = ParagraphStyle('title', fontSize=16, spaceAfter=10,
                                  textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    header_style = ParagraphStyle('header', fontSize=12, spaceAfter=6,
                                   textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=10, spaceAfter=6, fontName='Helvetica')

    story = []
    story.append(Paragraph("Adverse Event Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(Spacer(1, 0.5*cm))

    if report.is_sae:
        sae_style = ParagraphStyle('sae', fontSize=11, spaceAfter=8,
                                    textColor=colors.HexColor('#991b1b'), fontName='Helvetica-Bold')
        story.append(Paragraph("⚠ SERIOUS ADVERSE EVENT (SAE)", sae_style))
        if report.report_deadline:
            story.append(Paragraph(
                f"Reporting Deadline: {report.report_deadline.strftime('%Y-%m-%d')} "
                f"({report.days_until_deadline()}days remaining)",
                ParagraphStyle('deadline', fontSize=10, textColor=colors.HexColor('#991b1b'), fontName='Helvetica')
            ))
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Patient Information", header_style))
    story.append(_make_table([
        ['Field', 'Value'],
        ['Patient Code', report.patient_code],
        ['Age', str(report.age) if report.age else 'N/A'],
        ['Sex', 'Female' if report.sex == 'F' else 'Male' if report.sex == 'M' else 'N/A'],
    ]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Drug Information", header_style))
    story.append(_make_table([
        ['Field', 'Value'],
        ['Drug Name', report.drugname],
        ['Dose', report.dose or 'N/A'],
        ['Route', report.route or 'N/A'],
    ]))
    story.append(Spacer(1, 0.4*cm))

    grade_labels = {1:'Grade 1 (Mild)', 2:'Grade 2 (Moderate)', 3:'Grade 3 (Severe)',
                    4:'Grade 4 (Life-threatening)', 5:'Grade 5 (Death)'}
    story.append(Paragraph("Adverse Event Details", header_style))
    story.append(_make_table([
        ['Field', 'Value'],
        ['AE Term (MedDRA PT)', report.ae_term],
        ['CTCAE Grade', grade_labels.get(report.ctcae_grade, 'N/A')],
        ['SAE', 'YES' if report.is_sae else 'NO'],
        ['SAE Category', report.sae_category or 'N/A'],
        ['Causality', report.causality or 'N/A'],
        ['Onset Date', report.ae_start_date.strftime('%Y-%m-%d') if report.ae_start_date else 'N/A'],
        ['End Date', report.ae_end_date.strftime('%Y-%m-%d') if report.ae_end_date else 'Ongoing'],
        ['Action Taken', report.action_taken or 'N/A'],
        ['Outcome', report.outcome or 'N/A'],
    ]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Reporting Information", header_style))
    story.append(_make_table([
        ['Field', 'Value'],
        ['Report Date', report.reported_at.strftime('%Y-%m-%d %H:%M')],
        ['Deadline', report.report_deadline.strftime('%Y-%m-%d') if report.report_deadline else 'N/A'],
        ['Status', 'Submitted' if report.is_submitted else 'Pending'],
    ]))

    if report.notes:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("Notes", header_style))
        story.append(Paragraph(report.notes, sub_style))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'AE_{report.patient_code}_{report.id}.pdf',
                     mimetype='application/pdf')

@ae.route('/api/ae/<int:ae_id>/e2b')
def ae_e2b(ae_id):
    report = AEReport.query.get_or_404(ae_id)

    sex_code = '1' if report.sex == 'M' else '2' if report.sex == 'F' else '0'
    outcome_map = {
        '회복': '1', '회복중': '2', '후유증 지속': '3',
        '지속중': '4', '사망': '5', '불명': '6'
    }
    outcome_code = outcome_map.get(report.outcome or '', '6')

    sae_flags = {
        'seriousnessother': '1',
        'seriousnesshospitalization': '1' if report.sae_category == '입원' else '0',
        'seriousnesslifethreatening': '1' if report.sae_category == '생명위협' else '0',
        'seriousnessdisabling': '1' if report.sae_category == '영구장애' else '0',
        'seriousnesscongenitalanomali': '1' if report.sae_category == '선천성이상' else '0',
        'seriousnessdeath': '1' if report.sae_category == '사망' else '0',
    }

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- ICH E2B(R3) Adverse Event Report -->
<!-- Generated by Pharma Risk Analyzer -->
<!-- Report ID: AE-{report.id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} -->
<ichicsr lang="en">
  <ichicsrmessageheader>
    <messagetype>ichicsr</messagetype>
    <messageformatversion>2.1</messageformatversion>
    <messagenumb>AE-{report.id:06d}</messagenumb>
    <messagesenderidentifier>PHARMA-RISK-ANALYZER</messagesenderidentifier>
    <messagereceiveridentifier>REGULATORY-AUTHORITY</messagereceiveridentifier>
    <messagedateformat>204</messagedateformat>
    <messagedate>{datetime.now().strftime('%Y%m%d%H%M%S')}</messagedate>
  </ichicsrmessageheader>
  <safetyreport>
    <safetyreportid>AE-{report.id:06d}</safetyreportid>
    <primarysourcecountry>KR</primarysourcecountry>
    <occurcountry>KR</occurcountry>
    <transmissiondateformat>102</transmissiondateformat>
    <transmissiondate>{report.reported_at.strftime('%Y%m%d')}</transmissiondate>
    <serious>{'1' if report.is_sae else '2'}</serious>
    {''.join(f'    <{k}>{v}</{k}>\n' for k, v in sae_flags.items())}
    <receivedateformat>102</receivedateformat>
    <receivedate>{report.reported_at.strftime('%Y%m%d')}</receivedate>
    <patient>
      <patientinitial>{report.patient_code}</patientinitial>
      {'<patientagegroup>' + str(report.age) + '</patientagegroup>' if report.age else ''}
      <patientsex>{sex_code}</patientsex>
    </patient>
    <drug>
      <drugcharacterization>1</drugcharacterization>
      <medicinalproduct>{report.drugname}</medicinalproduct>
      {'<drugdosagetext>' + report.dose + '</drugdosagetext>' if report.dose else ''}
      {'<drugroute>' + report.route + '</drugroute>' if report.route else ''}
      <drugindication>UNKNOWN</drugindication>
    </drug>
    <reaction>
      <primarysourcereaction>{report.ae_term}</primarysourcereaction>
      <reactionmeddrapt>{report.ae_term}</reactionmeddrapt>
      {'<reactionstartdate>' + report.ae_start_date.strftime('%Y%m%d') + '</reactionstartdate>' if report.ae_start_date else ''}
      {'<reactionenddate>' + report.ae_end_date.strftime('%Y%m%d') + '</reactionenddate>' if report.ae_end_date else ''}
      <reactionoutcome>{outcome_code}</reactionoutcome>
    </reaction>
    <summary>
      <narrativeincludeclinical>
        Patient: {report.patient_code} | Drug: {report.drugname} | Reaction: {report.ae_term}
        CTCAE Grade: {report.ctcae_grade} | SAE: {'Yes' if report.is_sae else 'No'}
        Causality: {report.causality or 'Unknown'} | Outcome: {report.outcome or 'Unknown'}
      </narrativeincludeclinical>
    </summary>
  </safetyreport>
</ichicsr>'''

    buf = io.BytesIO(xml.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'E2B_AE-{report.id:06d}_{report.patient_code}.xml',
                     mimetype='application/xml')

@ae.route('/api/audit-trail')
def get_audit_trail():
    logs = AuditTrail.query.order_by(AuditTrail.timestamp.desc()).limit(50).all()
    return jsonify({'logs':[l.to_dict() for l in logs]})