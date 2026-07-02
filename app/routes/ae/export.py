"""
app/routes/ae/export.py
AE 문서 출력 - PDF Report, ICH E2B(R3) XML
"""
import io
from datetime import datetime, UTC
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm

from . import ae
from ._common import log_audit, _make_table
from app.models import AEReport


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
    story.append(Paragraph(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}", sub_style))
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
    log_audit('EXPORT', 'ae_reports', record_id=ae_id, reason='ICH E2B(R3) XML export')

    sex_code = '1' if report.sex == 'M' else '2' if report.sex == 'F' else '0'

    outcome_map = {
        '회복': '1', '회복후유증': '2', '미회복': '3',
        '지속중': '4', '사망': '5', '불명': '6',
        'recovered': '1', 'recovering': '2', 'not recovered': '3',
        'fatal': '5', 'unknown': '6'
    }
    outcome_code = outcome_map.get((report.outcome or '').lower(), '6')

    sae_category_map = {
        '사망': 'seriousnessdeath',
        '입원': 'seriousnesshospitalization',
        '생명위협': 'seriousnesslifethreatening',
        '영구장애': 'seriousnessdisabling',
        '선천성기형': 'seriousnesscongenitalanomali',
        '기타': 'seriousnessother',
        'death': 'seriousnessdeath',
        'hospitalization': 'seriousnesshospitalization',
        'life-threatening': 'seriousnesslifethreatening',
        'disability': 'seriousnessdisabling',
        'congenital anomaly': 'seriousnesscongenitalanomali',
        'other': 'seriousnessother',
    }
    sae_flags = {
        'seriousnessdeath': '0',
        'seriousnesshospitalization': '0',
        'seriousnesslifethreatening': '0',
        'seriousnessdisabling': '0',
        'seriousnesscongenitalanomali': '0',
        'seriousnessother': '1' if report.is_sae else '0',
    }
    if report.sae_category:
        key = sae_category_map.get(report.sae_category, None)
        if key:
            sae_flags[key] = '1'

    causality_map = {
        'Certain': '1', 'Probable': '2', 'Possible': '3',
        'Unlikely': '4', 'Unclassifiable': '5'
    }
    causality_code = causality_map.get(report.causality or '', '5')

    route_map = {
        '경구': '048', '정맥주사': '042', '근육주사': '058',
        '피하주사': '065', '흡입': '026', '외용': '003',
        'oral': '048', 'intravenous': '042', 'intramuscular': '058',
        'subcutaneous': '065', 'inhalation': '026', 'topical': '003',
    }
    route_code = route_map.get(report.route or '', '048')

    drug_duration = ''
    if report.ae_start_date and report.reported_at:
        days = (report.ae_start_date - report.reported_at.date()).days
        if days > 0:
            drug_duration = f'<drugtreatmentduration>{days}</drugtreatmentduration>\n        <drugtreatmentdurationunit>804</drugtreatmentdurationunit>'

    now_str = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    now_dt = datetime.now(UTC)

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- ICH E2B(R3) Individual Case Safety Report (ICSR) -->
<!-- Generated by Pharma Risk Analyzer v2.0 -->
<!-- Compliance: ICH E2B(R3) Guidelines (EMA/H/20665/04) -->
<!-- Report ID: AE-{report.id:06d} | Generated: {now_str} -->
<ichicsr lang="en">
  <ichicsrmessageheader>
    <messagetype>ichicsr</messagetype>
    <messageformatversion>2.1</messageformatversion>
    <messageformatrelease>2.0</messageformatrelease>
    <messagenumb>AE-{report.id:06d}-{now_dt.strftime('%Y%m%d')}</messagenumb>
    <messagesenderidentifier>PHARMA-RISK-ANALYZER-KR</messagesenderidentifier>
    <messagereceiveridentifier>MFDS-KR</messagereceiveridentifier>
    <messagedateformat>204</messagedateformat>
    <messagedate>{now_dt.strftime('%Y%m%d%H%M%S')}</messagedate>
  </ichicsrmessageheader>
  <safetyreport>
    <safetyreportid>AE-{report.id:06d}</safetyreportid>
    <safetyreportversion>1</safetyreportversion>
    <primarysourcecountry>KR</primarysourcecountry>
    <occurcountry>KR</occurcountry>
    <transmissiondateformat>102</transmissiondateformat>
    <transmissiondate>{report.reported_at.strftime('%Y%m%d')}</transmissiondate>
    <reporttype>1</reporttype>
    <serious>{'1' if report.is_sae else '2'}</serious>
    <seriousnessdeath>{sae_flags['seriousnessdeath']}</seriousnessdeath>
    <seriousnesslifethreatening>{sae_flags['seriousnesslifethreatening']}</seriousnesslifethreatening>
    <seriousnesshospitalization>{sae_flags['seriousnesshospitalization']}</seriousnesshospitalization>
    <seriousnessdisabling>{sae_flags['seriousnessdisabling']}</seriousnessdisabling>
    <seriousnesscongenitalanomali>{sae_flags['seriousnesscongenitalanomali']}</seriousnesscongenitalanomali>
    <seriousnessother>{sae_flags['seriousnessother']}</seriousnessother>
    <receivedateformat>102</receivedateformat>
    <receivedate>{report.reported_at.strftime('%Y%m%d')}</receivedate>
    <receiptdateformat>102</receiptdateformat>
    <receiptdate>{report.reported_at.strftime('%Y%m%d')}</receiptdate>
    <additionaldocument>2</additionaldocument>
    <fulfillexpeditecriteria>{'1' if report.is_sae else '2'}</fulfillexpeditecriteria>
    <primarysource>
      <reporterfamilyname>UNKNOWN</reporterfamilyname>
      <reportercountry>KR</reportercountry>
      <qualification>5</qualification>
    </primarysource>
    <sender>
      <sendertype>2</sendertype>
      <senderorganization>PHARMA-RISK-ANALYZER</senderorganization>
      <senderfamilyname>SYSTEM</senderfamilyname>
      <senderstreetaddress>KOREA</senderstreetaddress>
      <sendercountrycode>KR</sendercountrycode>
    </sender>
    <receiver>
      <receivertype>6</receivertype>
      <receiverorganization>MFDS</receiverorganization>
      <receivercountrycode>KR</receivercountrycode>
    </receiver>
    <patient>
      <patientinitial>{report.patient_code}</patientinitial>
      {'<patientbirthdateformat>102</patientbirthdateformat>' if report.age else ''}
      {'<patientagegroup>' + str(int(report.age // 10 * 10)) + '</patientagegroup>' if report.age else ''}
      {'<patientage>' + str(report.age) + '</patientage>' if report.age else ''}
      {'<patientageunit>801</patientageunit>' if report.age else ''}
      <patientsex>{sex_code}</patientsex>
      <drug>
        <drugcharacterization>1</drugcharacterization>
        <medicinalproduct>{report.drugname}</medicinalproduct>
        {'<drugdosagetext>' + report.dose + '</drugdosagetext>' if report.dose else ''}
        {'<drugdosageform>UNKNOWN</drugdosageform>' if report.dose else ''}
        <drugroute>{route_code}</drugroute>
        {'<drugstartdateformat>102</drugstartdateformat>' if report.ae_start_date else ''}
        {drug_duration}
        <drugindication>{report.ae_term}</drugindication>
        <drugactionindication>1</drugactionindication>
        <actiondrug>{'1' if report.action_taken and '중단' in report.action_taken else '6'}</actiondrug>
      </drug>
      <reaction>
        <primarysourcereaction>{report.ae_term}</primarysourcereaction>
        <reactionmeddraversionpt>26.1</reactionmeddraversionpt>
        <reactionmeddrapt>{report.ae_term}</reactionmeddrapt>
        {'<reactionstartdateformat>102</reactionstartdateformat>' if report.ae_start_date else ''}
        {'<reactionstartdate>' + report.ae_start_date.strftime('%Y%m%d') + '</reactionstartdate>' if report.ae_start_date else ''}
        {'<reactionenddateformat>102</reactionenddateformat>' if report.ae_end_date else ''}
        {'<reactionenddate>' + report.ae_end_date.strftime('%Y%m%d') + '</reactionenddate>' if report.ae_end_date else ''}
        <reactionoutcome>{outcome_code}</reactionoutcome>
      </reaction>
      <resultstest>
        <testname>CTCAE Grade</testname>
        <testresult>{report.ctcae_grade or 'N/A'}</testresult>
      </resultstest>
      <summary>
        <narrativeincludeclinical>
Case Summary (ICH E2B R3):
Patient Identifier: {report.patient_code}
Sex: {'Male' if report.sex == 'M' else 'Female' if report.sex == 'F' else 'Unknown'}
{'Age: ' + str(report.age) + ' years' if report.age else 'Age: Unknown'}

Suspect Drug: {report.drugname}
{'Dose: ' + report.dose if report.dose else ''}
{'Route: ' + report.route if report.route else ''}

Adverse Event: {report.ae_term}
CTCAE Grade: {report.ctcae_grade or 'Unknown'}
SAE: {'Yes - ' + (report.sae_category or '') if report.is_sae else 'No'}
Causality: {report.causality or 'Unknown'}
{'Onset: ' + report.ae_start_date.strftime('%Y-%m-%d') if report.ae_start_date else ''}
{'Resolution: ' + report.ae_end_date.strftime('%Y-%m-%d') if report.ae_end_date else 'Ongoing'}
Action Taken: {report.action_taken or 'Unknown'}
Outcome: {report.outcome or 'Unknown'}

{'Notes: ' + report.notes if report.notes else ''}

Generated by Pharma Risk Analyzer | ICH E2B(R3) Compliant Format
        </narrativeincludeclinical>
        <senderdiagnosis>{report.ae_term}</senderdiagnosis>
        <senderdiagnosiscode>{causality_code}</senderdiagnosiscode>
      </summary>
    </patient>
  </safetyreport>
</ichicsr>'''

    buf = io.BytesIO(xml.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'E2B_AE-{report.id:06d}_{report.patient_code}_{now_dt.strftime("%Y%m%d")}.xml',
                     mimetype='application/xml')