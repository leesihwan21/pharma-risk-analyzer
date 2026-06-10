import io
import os
import pickle
import pandas as pd
import plotly
import plotly.express as px
import json
import requests as http_requests

from flask import Blueprint, render_template, jsonify, request, send_file
from flask_login import login_required, current_user
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from app import cache
from app.models import db, DrugSearch, FavoriteDrug, PredictionLog

drug = Blueprint('drug', __name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ml')

def load_df():
    return pd.read_csv(DATA_PATH)

def load_model():
    model = pickle.load(open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb'))
    le_drug = pickle.load(open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'rb'))
    le_reac = pickle.load(open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'rb'))
    return model, le_drug, le_reac

@drug.route('/api/search/<drugname>')
@cache.cached(timeout=120)
def search_drug(drugname):
    df = load_df()
    result = df[df['drugname'].str.upper() == drugname.upper()]

    if len(result) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname.upper()}'}), 404

    top_reac = result['pt'].value_counts().head(10).reset_index()
    top_reac.columns = ['reaction', 'count']
    age_data = result['age'].dropna()
    age_avg = round(float(age_data.mean()), 1) if len(age_data) > 0 else 0
    sex_counts = result['sex'].value_counts().to_dict()
    outc_counts = result['outc_cod'].value_counts().to_dict()

    try:
        log = DrugSearch(drugname=drugname.upper(), total_reports=len(result), age_avg=age_avg)
        db.session.add(log)
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({
        'drug': drugname.upper(),
        'total_reports': len(result),
        'age_avg': age_avg,
        'sex_distribution': sex_counts,
        'outcome_distribution': outc_counts,
        'top_reactions': top_reac.to_dict(orient='records')
    })

@drug.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    drugname = data.get('drugname', '').upper()
    reaction = data.get('reaction', '').upper()
    sex = data.get('sex', 'F')
    age = float(data.get('age', 50))

    model, le_drug, le_reac = load_model()

    if drugname not in le_drug.classes_:
        return jsonify({'error': f'알 수 없는 약물: {drugname}'}), 400
    if reaction not in le_reac.classes_:
        return jsonify({'error': f'알 수 없는 부작용: {reaction}'}), 400

    drug_enc = le_drug.transform([drugname])[0]
    reac_enc = le_reac.transform([reaction])[0]
    sex_enc = 0 if sex == 'F' else 1

    risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
    drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
    reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
    combo_key = f"{drug_enc}_{reac_enc}"
    combo_risk_rate = risk_rates['combo_risk'].get(combo_key, 0.5)

    X = [[drug_enc, reac_enc, sex_enc, age, drug_risk_rate, reac_risk_rate, combo_risk_rate]]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    try:
        log = PredictionLog(
            drugname=drugname.upper(), reaction=reaction.upper(),
            age=age, sex=sex, risk=int(pred),
            safe_prob=round(float(prob[0]) * 100, 1),
            risk_prob=round(float(prob[1]) * 100, 1)
        )
        db.session.add(log)
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({
        'drug': drugname, 'reaction': reaction, 'risk': int(pred),
        'risk_label': '⚠️ 고위험 (입원/사망 가능성)' if pred == 1 else '✅ 저위험',
        'probability': {
            'safe': round(float(prob[0]) * 100, 1),
            'risk': round(float(prob[1]) * 100, 1)
        }
    })

@drug.route('/api/combo', methods=['POST'])
def combo_risk():
    data = request.get_json()
    drug1 = data.get('drug1', '').upper()
    drug2 = data.get('drug2', '').upper()
    age = float(data.get('age', 50))
    sex = data.get('sex', 'F')

    model, le_drug, le_reac = load_model()
    risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))

    if drug1 not in le_drug.classes_:
        return jsonify({'error': f'알 수 없는 약물: {drug1}'}), 400
    if drug2 not in le_drug.classes_:
        return jsonify({'error': f'알 수 없는 약물: {drug2}'}), 400

    sex_enc = 0 if sex == 'F' else 1
    results = []

    for d in [drug1, drug2]:
        drug_enc = le_drug.transform([d])[0]
        drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
        df = load_df()
        top_reacs = df[df['drugname'].str.upper() == d]['pt'].value_counts().head(5).index.tolist()

        drug_results = []
        for reac in top_reacs:
            if reac not in le_reac.classes_:
                continue
            reac_enc = le_reac.transform([reac])[0]
            reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
            combo_risk_rate = risk_rates['combo_risk'].get(f"{drug_enc}_{reac_enc}", 0.5)
            X = [[drug_enc, reac_enc, sex_enc, age, drug_risk_rate, reac_risk_rate, combo_risk_rate]]
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            drug_results.append({
                'reaction': reac, 'risk': int(pred),
                'risk_label': '⚠️ 고위험' if pred == 1 else '✅ 저위험',
                'risk_prob': round(float(prob[1]) * 100, 1)
            })

        results.append({
            'drug': d,
            'drug_risk_rate': round(drug_risk_rate * 100, 1),
            'reactions': drug_results
        })

    return jsonify({'results': results})

@drug.route('/api/autocomplete/<keyword>')
@cache.cached(timeout=600)
def autocomplete(keyword):
    df = load_df()
    keyword = keyword.upper()
    drugs = df['drugname'].str.upper().unique()
    matches = [d for d in drugs if d.startswith(keyword)][:10]
    return jsonify({'suggestions': sorted(matches)})

@drug.route('/drug/<drugname>')
@cache.cached(timeout=300)
def drug_detail(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return "약물을 찾을 수 없어요", 404

    top_reac = result['pt'].value_counts().head(15).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count',
                  title=f'{drugname} - 부작용 TOP 15',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    age_data = result['age'].dropna()
    fig2 = px.histogram(age_data, nbins=20,
                        title=f'{drugname} - 나이 분포',
                        color_discrete_sequence=['#38bdf8'])
    fig2.update_layout(template='plotly_dark', height=420)

    sex_counts = result['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count',
                  title=f'{drugname} - 성별 분포', template='plotly_dark')
    fig3.update_layout(height=420)

    outc_counts = result['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count',
                  title=f'{drugname} - 결과 분포',
                  color='count', color_continuous_scale='Reds')
    fig4.update_layout(template='plotly_dark', height=420)

    stats = {
        'total': len(result),
        'age_avg': round(float(age_data.mean()), 1) if len(age_data) > 0 else 0,
        'age_min': int(age_data.min()) if len(age_data) > 0 else 0,
        'age_max': int(age_data.max()) if len(age_data) > 0 else 0,
    }
    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        'chart4': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder),
    }
    return render_template('drug_detail.html', drugname=drugname, stats=stats, charts=charts)

@drug.route('/api/report/<drugname>')
def generate_report(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '약물을 찾을 수 없어요'}), 404

    font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'NanumGothic.ttf')
    try:
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        korean_font = 'NanumGothic'
    except:
        korean_font = 'Helvetica'

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', fontSize=18, spaceAfter=12,
                                  textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=11, spaceAfter=8,
                                textColor=colors.HexColor('#374151'), fontName='Helvetica')
    header_style = ParagraphStyle('header', fontSize=13, spaceAfter=6,
                                   textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    story = []
    story.append(Paragraph(f"Pharma Risk Analyzer - Drug Report", title_style))
    story.append(Paragraph(f"Drug: {drugname}", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(Spacer(1, 0.5*cm))

    age_data = result['age'].dropna()
    story.append(Paragraph("Basic Statistics", header_style))
    stats_data = [
        ['Metric', 'Value'],
        ['Total Reports', str(len(result))],
        ['Average Age', f"{round(float(age_data.mean()), 1) if len(age_data) > 0 else 'N/A'}"],
        ['Age Range', f"{int(age_data.min())} ~ {int(age_data.max())}" if len(age_data) > 0 else 'N/A'],
        ['Female (%)', f"{round(len(result[result['sex']=='F']) / len(result) * 100, 1)}%"],
        ['Male (%)', f"{round(len(result[result['sex']=='M']) / len(result) * 100, 1)}%"],
    ]
    t = Table(stats_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a56db')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Top 10 Adverse Reactions", header_style))
    top_reac = result['pt'].value_counts().head(10)
    reac_data = [['Rank', 'Adverse Reaction', 'Count', 'Percentage']]
    for i, (reac, cnt) in enumerate(top_reac.items(), 1):
        pct = round(cnt / len(result) * 100, 1)
        reac_data.append([str(i), reac, str(cnt), f"{pct}%"])
    t2 = Table(reac_data, colWidths=[2*cm, 9*cm, 3*cm, 3*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a56db')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    try:
        top_reac_list = result['pt'].value_counts().head(5).index.tolist()
        death_cnt = len(result[result['outc_cod']=='DE'])
        hosp_cnt = len(result[result['outc_cod']=='HO'])

        prompt = f"""반드시 한국어로만 답하세요. 영어를 절대 사용하지 마세요.

다음 FDA FAERS 약물 데이터를 3-4문장으로 한국어 요약해주세요:
약물명: {drugname}
총 부작용 보고건수: {len(result)}건
주요 부작용 TOP5: {', '.join(top_reac_list)}
사망 보고: {death_cnt}건
입원 보고: {hosp_cnt}건
평균 나이: {round(float(age_data.mean()), 1) if len(age_data) > 0 else 'N/A'}세"""

        response = http_requests.post('http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt, 'stream': False}, timeout=60)
        ai_summary = response.json().get('response', '')

        if ai_summary:
            ai_style = ParagraphStyle('ai', fontSize=10, spaceAfter=6,
                                       textColor=colors.HexColor('#374151'),
                                       fontName=korean_font, leading=14)
            story.append(Paragraph("AI 자동 요약 (powered by Llama3.2)", header_style))
            story.append(Paragraph(ai_summary, ai_style))
    except Exception as e:
        pass

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'{drugname}_report.pdf',
                     mimetype='application/pdf')

@drug.route('/api/favorite/<drugname>', methods=['POST'])
def add_favorite(drugname):
    try:
        fav = FavoriteDrug(drugname=drugname.upper())
        db.session.add(fav)
        db.session.commit()
        return jsonify({'message': f'{drugname} 즐겨찾기 추가!'})
    except:
        db.session.rollback()
        return jsonify({'message': '이미 즐겨찾기에 있어요'})

@drug.route('/api/favorites')
def get_favorites():
    favs = FavoriteDrug.query.order_by(FavoriteDrug.added_at.desc()).all()
    return jsonify({'favorites': [f.to_dict() for f in favs]})

@drug.route('/api/history')
def get_history():
    searches = DrugSearch.query.order_by(DrugSearch.searched_at.desc()).limit(20).all()
    predictions = PredictionLog.query.order_by(PredictionLog.predicted_at.desc()).limit(20).all()
    return jsonify({
        'searches': [s.to_dict() for s in searches],
        'predictions': [p.to_dict() for p in predictions]
    })

@drug.route('/api/network/<drugname>')
@cache.cached(timeout=300)
def drug_network(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '약물을 찾을 수 없어요'}), 404

    top_reac = result['pt'].value_counts().head(15)
    nodes = [{'id': drugname, 'type': 'drug', 'size': 30}]
    edges = []
    for reac, cnt in top_reac.items():
        nodes.append({'id': reac, 'type': 'reaction', 'size': max(8, min(20, cnt // 10))})
        edges.append({'source': drugname, 'target': reac, 'weight': int(cnt)})
    return jsonify({'nodes': nodes, 'edges': edges})

@drug.route('/api/timeline/<drugname>')
@cache.cached(timeout=300)
def drug_timeline(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '약물을 찾을 수 없어요'}), 404

    result = result.copy()
    result['year'] = pd.to_datetime(result['event_dt'], errors='coerce').dt.year
    yearly = result.groupby('year').size().reset_index(name='count')
    yearly = yearly.dropna(subset=['year'])
    yearly['year'] = yearly['year'].astype(int)
    yearly = yearly[yearly['year'] >= 2000]
    return jsonify({'drugname': drugname, 'timeline': yearly.to_dict(orient='records')})

@drug.route('/compare')
def compare():
    return render_template('compare.html')

@drug.route('/api/compare')
def compare_drugs():
    drug1 = request.args.get('drug1', '').upper()
    drug2 = request.args.get('drug2', '').upper()

    if not drug1 or not drug2:
        return jsonify({'error': '약물 2개를 입력하세요'}), 400

    df = load_df()
    result1 = df[df['drugname'].str.upper() == drug1]
    result2 = df[df['drugname'].str.upper() == drug2]

    if len(result1) == 0:
        return jsonify({'error': f'{drug1} 약물을 찾을 수 없어요'}), 404
    if len(result2) == 0:
        return jsonify({'error': f'{drug2} 약물을 찾을 수 없어요'}), 404

    def get_stats(result, name):
        age_data = result['age'].dropna()
        top_reac = result['pt'].value_counts().head(10).reset_index()
        top_reac.columns = ['reaction', 'count']
        return {
            'drug': name,
            'total_reports': len(result),
            'age_avg': round(float(age_data.mean()), 1) if len(age_data) > 0 else 0,
            'female_pct': round(len(result[result['sex']=='F']) / len(result) * 100, 1),
            'male_pct': round(len(result[result['sex']=='M']) / len(result) * 100, 1),
            'death_cnt': len(result[result['outc_cod']=='DE']),
            'hosp_cnt': len(result[result['outc_cod']=='HO']),
            'top_reactions': top_reac.to_dict(orient='records')
        }

    return jsonify({'drug1': get_stats(result1, drug1), 'drug2': get_stats(result2, drug2)})

@drug.route('/api/filter')
def filter_data():
    d = request.args.get('drug', '').upper()
    sex = request.args.get('sex', '')
    age_min = request.args.get('age_min', 0, type=int)
    age_max = request.args.get('age_max', 120, type=int)
    outcome = request.args.get('outcome', '')
    country = request.args.get('country', '')

    df = load_df()
    if d: df = df[df['drugname'].str.upper() == d]
    if sex: df = df[df['sex'] == sex]
    if age_min: df = df[df['age'] >= age_min]
    if age_max < 120: df = df[df['age'] <= age_max]
    if outcome: df = df[df['outc_cod'] == outcome]
    if country: df = df[df['reporter_country'].str.upper() == country.upper()]

    if len(df) == 0:
        return jsonify({'error': '조건에 맞는 데이터가 없어요'}), 404

    top_reac = df['pt'].value_counts().head(10).reset_index()
    top_reac.columns = ['reaction', 'count']
    age_data = df['age'].dropna()

    return jsonify({
        'total': len(df),
        'age_avg': round(float(age_data.mean()), 1) if len(age_data) > 0 else 0,
        'female_pct': round(len(df[df['sex']=='F']) / len(df) * 100, 1),
        'male_pct': round(len(df[df['sex']=='M']) / len(df) * 100, 1),
        'death_cnt': len(df[df['outc_cod']=='DE']),
        'hosp_cnt': len(df[df['outc_cod']=='HO']),
        'top_reactions': top_reac.to_dict(orient='records')
    })

@drug.route('/filter')
def filter_page():
    return render_template('filter.html')

@drug.route('/api/send_report/<drugname>', methods=['POST'])
@login_required
def send_report_email(drugname):
    from flask_mail import Message
    from app import mail
    data = request.get_json()
    email = data.get('email', '')
    if not email:
        return jsonify({'error': '이메일을 입력하세요'}), 400

    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]
    if len(result) == 0:
        return jsonify({'error': '약물을 찾을 수 없어요'}), 404

    top_reac = result['pt'].value_counts().head(5)
    age_data = result['age'].dropna()
    body = f"""
Pharma Risk Analyzer - Drug Report
====================================
Drug: {drugname}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Basic Statistics:
- Total Reports: {len(result):,}
- Average Age: {round(float(age_data.mean()), 1) if len(age_data) > 0 else 'N/A'}
- Age Range: {int(age_data.min())} ~ {int(age_data.max())}

Top 5 Adverse Reactions:
"""
    for i, (reac, cnt) in enumerate(top_reac.items(), 1):
        body += f"{i}. {reac}: {cnt}건\n"
    body += "\n\nPowered by Pharma Risk Analyzer"

    try:
        msg = Message(
            subject=f'[Pharma Risk Analyzer] {drugname} 분석 리포트',
            recipients=[email],
            body=body
        )
        mail.send(msg)
        return jsonify({'message': f'{email}로 리포트를 전송했습니다!'})
    except Exception as e:
        return jsonify({'error': f'메일 전송 실패: {str(e)}'}), 500

@drug.route('/api/llm_explain', methods=['POST'])
def llm_explain():
    data = request.get_json()
    drug_name = data.get('drug', '')
    reaction = data.get('reaction', '')
    risk_label = data.get('risk_label', '')
    safe_prob = data.get('safe_prob', 0)
    risk_prob = data.get('risk_prob', 0)

    prompt = f"""반드시 한국어로만 답하세요. 다른 언어는 절대 사용하지 마세요.

약물 부작용 분석 결과를 한국어로 3-4문장 설명해주세요:
약물: {drug_name}
부작용: {reaction}
AI 판정: {risk_label}
안전 확률: {safe_prob}%
위험 확률: {risk_prob}%"""

    try:
        response = http_requests.post('http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt, 'stream': False}, timeout=60)
        result = response.json()
        return jsonify({'explanation': result['response']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@drug.route('/api/safety_report/<drugname>')
def safety_report(drugname):
    drugname = drugname.upper()
    df = load_df()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname}'}), 404

    # FDA FAERS 통계
    age_data = result['age'].dropna()
    top_reac = result['pt'].value_counts().head(5).index.tolist()
    death_cnt = len(result[result['outc_cod'] == 'DE'])
    hosp_cnt = len(result[result['outc_cod'] == 'HO'])
    total = len(result)
    age_avg = round(float(age_data.mean()), 1) if len(age_data) > 0 else 0

    # PubMed 논문 검색
    pubmed_abstracts = ''
    ids = []
    try:
        search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
        search_params = {
            'db': 'pubmed',
            'term': f'{drugname} adverse event safety pharmacovigilance',
            'retmax': 5,
            'retmode': 'json',
            'sort': 'relevance'
        }
        search_res = http_requests.get(search_url, params=search_params, timeout=10)
        ids = search_res.json()['esearchresult']['idlist']

        if ids:
            fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'rettype': 'abstract',
                'retmode': 'text'
            }
            fetch_res = http_requests.get(fetch_url, params=fetch_params, timeout=10)
            pubmed_abstracts = fetch_res.text[:6000]
    except:
        pubmed_abstracts = '논문 검색 실패'

    # Ollama llama3.2로 안전성 리포트 생성
    prompt = f"""반드시 한국어로만 작성하세요. 다른 언어 절대 금지.

당신은 임상약물감시(Pharmacovigilance) 전문가입니다.
아래 FDA FAERS 실제 데이터와 PubMed 논문을 분석하여 상세한 약물 안전성 리포트를 작성하세요.
각 섹션을 3~4문장씩 구체적인 수치와 함께 작성하세요.

[FDA FAERS 실제 데이터]
- 약물명: {drugname}
- 총 부작용 보고: {total}건
- 주요 부작용 TOP5: {', '.join(top_reac)}
- 사망 보고: {death_cnt}건 (전체의 {round(death_cnt/total*100,1) if total > 0 else 0}%)
- 입원 보고: {hosp_cnt}건 (전체의 {round(hosp_cnt/total*100,1) if total > 0 else 0}%)
- 평균 환자 나이: {age_avg}세

[PubMed 관련 논문 초록]
{pubmed_abstracts}

아래 형식으로 각 섹션을 작성하세요:

## 1. 약물 개요
(작용기전, 주요 적응증 설명)

## 2. FDA FAERS 부작용 분석
(위 통계 수치를 구체적으로 인용하며 분석)

## 3. PubMed 논문 기반 안전성 근거
(논문에서 확인된 안전성 관련 내용)

## 4. 고위험군 및 주의사항
(특별히 주의가 필요한 환자군, 모니터링 항목)

## 5. 임상적 권고사항
(의료진을 위한 실질적 권고사항)

## 6. 결론
(종합적 안전성 평가)"""

    try:
        response = http_requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt, 'stream': False},
            timeout=120
        )
        report_text = response.json().get('response', '리포트 생성 실패')
    except Exception as e:
        report_text = f'Ollama 연결 실패: {str(e)}'

    return jsonify({
        'drug': drugname,
        'stats': {
            'total_reports': total,
            'age_avg': age_avg,
            'death_cnt': death_cnt,
            'hosp_cnt': hosp_cnt,
            'top_reactions': top_reac
        },
        'pubmed_count': len(ids),
        'report': report_text
    })
