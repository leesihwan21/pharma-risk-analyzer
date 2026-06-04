from flask import Blueprint, render_template, jsonify, request, send_file
import pandas as pd
import plotly.express as px
import plotly
import json
import os
import pycountry
from flask import Response
from flask_mail import Message
from flask_restx import Api, Resource, fields
from app import cache, limiter, mail
from datetime import datetime, timedelta, date
from flask_login import login_user, logout_user, login_required, current_user
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from app.models import db, DrugSearch, FavoriteDrug, PredictionLog, User, AEReport
from werkzeug.security import generate_password_hash, check_password_hash
import io
import math

main = Blueprint('main', __name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'processed', 'processed_faers.csv')

KOREA_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'raw', 'korea_adr.csv')

def load_df():
    return pd.read_csv(DATA_PATH)

def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

# ── 페이지 라우트 ──────────────────────────────────────────────

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@cache.cached(timeout=300)
def dashboard():
    df = load_df()

    top_reac = df['pt'].value_counts().head(20).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count', title='Top 20 Adverse Reactions',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    top_drug = df['drugname'].value_counts().head(20).reset_index()
    top_drug.columns = ['drug', 'count']
    fig2 = px.bar(top_drug, x='drug', y='count', title='Top 20 Drugs by Report Count',
                  color='count', color_continuous_scale='Reds')
    fig2.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    sex_counts = df['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count', title='Gender Distribution', template='plotly_dark')
    fig3.update_layout(height=420)

    outc_counts = df['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count', title='Outcome Distribution',
                  color='count', color_continuous_scale='Greens')
    fig4.update_layout(template='plotly_dark', height=420)

    df['age_group'] = pd.cut(df['age'], bins=[0,18,30,45,60,75,120],
        labels=['0-18','19-30','31-45','46-60','61-75','76+'])
    age_counts = df['age_group'].value_counts().sort_index().reset_index()
    age_counts.columns = ['age_group', 'count']
    fig5 = px.bar(age_counts, x='age_group', y='count', title='나이대별 부작용 보고 건수',
                  color='count', color_continuous_scale='Purples')
    fig5.update_layout(template='plotly_dark', height=420)

    country_counts = df['reporter_country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts['iso3'] = country_counts['country'].apply(alpha2_to_alpha3)
    country_counts = country_counts.dropna(subset=['iso3'])
    fig6 = px.scatter_geo(country_counts, locations='iso3', size='count', hover_name='country',
        title='국가별 부작용 보고 건수', color='count', color_continuous_scale='Reds', projection='natural earth')
    fig6.update_layout(template='plotly_dark', height=420,
        geo=dict(showframe=False, showcoastlines=True, showland=True, landcolor='#1e293b', bgcolor='#0f172a'))

    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        'chart4': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder),
        'chart5': json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder),
        'chart6': json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder),
    }
    return render_template('dashboard.html', charts=charts)

@main.route('/compare')
def compare():
    return render_template('compare.html')

@main.route('/filter')
def filter_page():
    return render_template('filter.html')

@main.route('/korea')
def korea_dashboard():
    df = pd.read_csv(KOREA_DATA_PATH, encoding='cp949')
    fig1 = px.bar(df.head(10), x='연도별증상(2024)', y='연도별보고건수(2024)',
        title='한국 2024년 Top 10 이상사례', color='연도별보고건수(2024)', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    years = ['2019','2020','2021','2022','2023','2024']
    top5 = df.head(5)['연도별증상(2024)'].tolist()
    fig2 = px.line(title='한국 Top 5 증상 연도별 트렌드')
    for symptom in top5:
        row = df[df['연도별증상(2024)'] == symptom]
        if len(row) == 0: continue
        counts = [int(row[f'연도별보고건수({y})'].values[0]) if f'연도별보고건수({y})' in df.columns else 0 for y in years]
        fig2.add_scatter(x=years, y=counts, name=symptom, mode='lines+markers')
    fig2.update_layout(template='plotly_dark', height=420)

    fig3 = px.bar(df.head(10), x='연도별증상(2024)', y=['연도별보고건수(2024)','연도별보고건수(2023)'],
        title='2024 vs 2023 Top 10 증상 비교', barmode='group',
        color_discrete_sequence=['#38bdf8','#a78bfa'])
    fig3.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
    }
    return render_template('korea.html', charts=charts)

@main.route('/drug/<drugname>')
@cache.cached(timeout=300)
def drug_detail(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]
    if len(result) == 0:
        return "약물을 찾을 수 없어요", 404

    top_reac = result['pt'].value_counts().head(15).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count', title=f'{drugname} - 부작용 TOP 15',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    age_data = result['age'].dropna()
    fig2 = px.histogram(age_data, nbins=20, title=f'{drugname} - 환자 나이 분포',
                        color_discrete_sequence=['#38bdf8'])
    fig2.update_layout(template='plotly_dark', height=420)

    sex_counts = result['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count', title=f'{drugname} - 성별 분포', template='plotly_dark')
    fig3.update_layout(height=420)

    outc_counts = result['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count', title=f'{drugname} - 결과 분포',
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

@main.route('/ae_manager')
def ae_manager():
    return render_template('ae_manager.html')

@main.route('/prr')
def prr_page():
    return render_template('prr.html')

# ── API 라우트 ─────────────────────────────────────────────────

@main.route('/api/search/<drugname>')
@cache.cached(timeout=300)
def search_drug(drugname):
    df = load_df()
    result = df[df['drugname'].str.upper() == drugname.upper()]
    if len(result) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname.upper()}'}), 404

    top_reac = result['pt'].value_counts().head(10).reset_index()
    top_reac.columns = ['reaction', 'count']
    age_data = result['age'].dropna()
    age_avg = round(float(age_data.mean()), 1) if len(age_data) > 0 else 0

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
        'sex_distribution': result['sex'].value_counts().to_dict(),
        'outcome_distribution': result['outc_cod'].value_counts().to_dict(),
        'top_reactions': top_reac.to_dict(orient='records')
    })

@main.route('/api/autocomplete/<keyword>')
@cache.cached(timeout=600)
def autocomplete(keyword):
    df = load_df()
    keyword = keyword.upper()
    drugs = df['drugname'].str.upper().unique()
    matches = [d for d in drugs if d.startswith(keyword)][:10]
    return jsonify({'suggestions': sorted(matches)})

@main.route('/api/compare')
def compare_drugs():
    drug1 = request.args.get('drug1', '').upper()
    drug2 = request.args.get('drug2', '').upper()
    if not drug1 or not drug2:
        return jsonify({'error': '약물 2개를 입력해주세요'}), 400

    df = load_df()
    result1 = df[df['drugname'].str.upper() == drug1]
    result2 = df[df['drugname'].str.upper() == drug2]

    if len(result1) == 0: return jsonify({'error': f'{drug1} 을 찾을 수 없어요'}), 404
    if len(result2) == 0: return jsonify({'error': f'{drug2} 을 찾을 수 없어요'}), 404

    def get_stats(result, name):
        age_data = result['age'].dropna()
        top_reac = result['pt'].value_counts().head(10).reset_index()
        top_reac.columns = ['reaction', 'count']
        return {
            'drug': name, 'total_reports': len(result),
            'age_avg': round(float(age_data.mean()), 1) if len(age_data) > 0 else 0,
            'female_pct': round(len(result[result['sex']=='F']) / len(result) * 100, 1),
            'male_pct': round(len(result[result['sex']=='M']) / len(result) * 100, 1),
            'death_cnt': len(result[result['outc_cod']=='DE']),
            'hosp_cnt': len(result[result['outc_cod']=='HO']),
            'top_reactions': top_reac.to_dict(orient='records')
        }
    return jsonify({'drug1': get_stats(result1, drug1), 'drug2': get_stats(result2, drug2)})

@main.route('/api/filter')
def filter_data():
    drug = request.args.get('drug', '').upper()
    sex = request.args.get('sex', '')
    age_min = request.args.get('age_min', 0, type=int)
    age_max = request.args.get('age_max', 120, type=int)
    outcome = request.args.get('outcome', '')
    country = request.args.get('country', '')

    df = load_df()
    if drug: df = df[df['drugname'].str.upper() == drug]
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

@main.route('/api/network/<drugname>')
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

@main.route('/api/report/<drugname>')
def generate_report(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]
    if len(result) == 0:
        return jsonify({'error': '약물을 찾을 수 없어요'}), 404

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', fontSize=18, spaceAfter=12, textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=11, spaceAfter=8, textColor=colors.HexColor('#374151'), fontName='Helvetica')
    header_style = ParagraphStyle('header', fontSize=13, spaceAfter=6, textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')

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
    t = _make_table(stats_data)
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Top 10 Adverse Reactions", header_style))
    top_reac = result['pt'].value_counts().head(10)
    reac_data = [['Rank', 'Adverse Reaction', 'Count', 'Percentage']]
    for i, (reac, cnt) in enumerate(top_reac.items(), 1):
        reac_data.append([str(i), reac, str(cnt), f"{round(cnt/len(result)*100,1)}%"])
    story.append(_make_table(reac_data))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f'{drugname}_report.pdf', mimetype='application/pdf')

@main.route('/api/favorite/<drugname>', methods=['POST'])
def add_favorite(drugname):
    try:
        fav = FavoriteDrug(drugname=drugname.upper())
        db.session.add(fav)
        db.session.commit()
        return jsonify({'message': f'{drugname} 즐겨찾기 추가!'})
    except:
        db.session.rollback()
        return jsonify({'message': '이미 즐겨찾기에 있어요'})

@main.route('/api/favorites')
def get_favorites():
    favs = FavoriteDrug.query.order_by(FavoriteDrug.added_at.desc()).all()
    return jsonify({'favorites': [f.to_dict() for f in favs]})

@main.route('/api/history')
def get_history():
    searches = DrugSearch.query.order_by(DrugSearch.searched_at.desc()).limit(20).all()
    predictions = PredictionLog.query.order_by(PredictionLog.predicted_at.desc()).limit(20).all()
    return jsonify({'searches': [s.to_dict() for s in searches], 'predictions': [p.to_dict() for p in predictions]})

@main.route('/api/me')
def me():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        email = data.get('email', '')
        password = data.get('password', '')
        if User.query.filter_by(username=username).first():
            return jsonify({'error': '이미 사용 중인 아이디예요'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': '이미 사용 중인 이메일이에요'}), 400
        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return jsonify({'message': '회원가입 성공!', 'username': username})
    return render_template('register.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': '아이디 또는 비밀번호가 틀려요'}), 401
        login_user(user)
        return jsonify({'message': f'{username}님 환영해요!', 'username': username})
    return render_template('login.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': '로그아웃됐어요'})

@main.route('/api/timeline/<drugname>')
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

# ── PRR ───────────────────────────────────────────────────────

@main.route('/api/prr/<drugname>')
@cache.cached(timeout=600)
def calculate_prr(drugname):
    df = load_df()
    drugname = drugname.upper()
    drug_reports = df[df['drugname'].str.upper() == drugname]
    if len(drug_reports) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname}'}), 404

    other_reports = df[df['drugname'].str.upper() != drugname]
    total_drug = len(drug_reports)
    total_other = len(other_reports)

    if total_other == 0:
        return jsonify({'error': '비교 데이터가 부족해요'}), 400

    top_reactions = drug_reports['pt'].value_counts().head(20).index.tolist()
    results = []
    for reac in top_reactions:
        a = len(drug_reports[drug_reports['pt'] == reac])
        b = total_drug
        c = len(other_reports[other_reports['pt'] == reac])
        d = total_other
        if c == 0 or b == 0: continue
        prr = (a / b) / (c / d)
        try:
            se = math.sqrt((1/a) - (1/b) + (1/c) - (1/d))
            prr_lower = math.exp(math.log(prr) - 1.96 * se)
            prr_upper = math.exp(math.log(prr) + 1.96 * se)
        except:
            prr_lower = prr_upper = prr
        is_signal = prr >= 2 and a >= 3
        results.append({
            'reaction': reac, 'drug_count': int(a), 'drug_total': int(b),
            'other_count': int(c), 'other_total': int(d),
            'drug_pct': round(a/b*100, 2), 'other_pct': round(c/d*100, 2),
            'prr': round(prr, 2), 'prr_lower': round(prr_lower, 2), 'prr_upper': round(prr_upper, 2),
            'is_signal': is_signal,
            'signal_level': ('🔴 강한 신호' if prr >= 5 and a >= 3 else '🟡 신호' if prr >= 2 and a >= 3 else '⚪ 비신호')
        })
    results.sort(key=lambda x: x['prr'], reverse=True)
    signal_count = sum(1 for r in results if r['is_signal'])
    strong_signal_count = sum(1 for r in results if r['prr'] >= 5 and r['drug_count'] >= 3)
    return jsonify({'drugname': drugname, 'total_reports': total_drug,
                    'signal_count': signal_count, 'strong_signal_count': strong_signal_count, 'results': results})

# ── AE Manager ────────────────────────────────────────────────

CTCAE_KEYWORDS = {
    5: ['death','fatal','사망'], 4: ['life-threatening','생명위협','icu','ventilat'],
    3: ['hospitali','severe','입원','중증'], 2: ['moderate','중등도','limiting'], 1: ['mild','경미','minor'],
}
SAE_KEYWORDS = ['사망','입원','생명위협','영구장애','선천성','death','hospitali','life-threatening','disability','congenital']

def auto_ctcae_grade(ae_term):
    term_lower = ae_term.lower()
    for grade in [5,4,3,2,1]:
        for kw in CTCAE_KEYWORDS[grade]:
            if kw in term_lower: return grade
    return 1

def auto_is_sae(ae_term, ctcae_grade):
    if ctcae_grade >= 3: return True
    return any(kw in ae_term.lower() for kw in SAE_KEYWORDS)

@main.route('/api/ae/list')
def ae_list():
    status_filter = request.args.get('status', '')
    sae_only = request.args.get('sae_only', 'false') == 'true'
    query = AEReport.query.order_by(AEReport.reported_at.desc())
    if sae_only: query = query.filter_by(is_sae=True)
    reports = query.all()
    result = [r.to_dict() for r in reports]
    if status_filter: result = [r for r in result if r['deadline_status'] == status_filter]
    all_reports = [r.to_dict() for r in AEReport.query.all()]
    summary = {
        'total': len(all_reports),
        'sae_count': sum(1 for r in all_reports if r['is_sae']),
        'overdue': sum(1 for r in all_reports if r['deadline_status'] == 'overdue'),
        'urgent': sum(1 for r in all_reports if r['deadline_status'] == 'urgent'),
        'submitted': sum(1 for r in all_reports if r['is_submitted']),
    }
    return jsonify({'reports': result, 'summary': summary})

@main.route('/api/ae/<int:ae_id>')
def ae_detail(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    return jsonify(report.to_dict())

@main.route('/api/ae/create', methods=['POST'])
def ae_create():
    data = request.get_json()
    if not data.get('patient_code') or not data.get('drugname') or not data.get('ae_term'):
        return jsonify({'error': '환자코드, 약물명, AE 용어는 필수예요'}), 400
    ae_term = data.get('ae_term', '')
    ctcae_grade = int(data.get('ctcae_grade') or auto_ctcae_grade(ae_term))
    is_sae_input = data.get('is_sae')
    is_sae = bool(is_sae_input) if is_sae_input is not None else auto_is_sae(ae_term, ctcae_grade)
    report_deadline = datetime.utcnow() + timedelta(days=15) if is_sae else None
    ae_start = ae_end = None
    try:
        if data.get('ae_start_date'): ae_start = datetime.strptime(data['ae_start_date'], '%Y-%m-%d').date()
        if data.get('ae_end_date'): ae_end = datetime.strptime(data['ae_end_date'], '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': '날짜 형식이 올바르지 않아요 (YYYY-MM-DD)'}), 400

    report = AEReport(
        patient_code=data.get('patient_code','').upper(), age=float(data['age']) if data.get('age') else None,
        sex=data.get('sex',''), drugname=data.get('drugname','').upper(), dose=data.get('dose',''),
        route=data.get('route',''), ae_term=ae_term, ae_start_date=ae_start, ae_end_date=ae_end,
        ctcae_grade=ctcae_grade, is_sae=is_sae, sae_category=data.get('sae_category',''),
        causality=data.get('causality',''), action_taken=data.get('action_taken',''),
        outcome=data.get('outcome',''), report_deadline=report_deadline, is_submitted=False, notes=data.get('notes',''),
    )
    try:
        db.session.add(report)
        db.session.commit()
        return jsonify({'message': 'AE 보고서가 등록됐어요', 'id': report.id, 'is_sae': is_sae,
                        'ctcae_grade': ctcae_grade,
                        'report_deadline': report_deadline.strftime('%Y-%m-%d') if report_deadline else None}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/api/ae/<int:ae_id>/update', methods=['POST'])
def ae_update(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    data = request.get_json()
    for f in ['ae_term','ctcae_grade','is_sae','sae_category','causality','action_taken','outcome','notes','dose','route','age','sex']:
        if f in data: setattr(report, f, data[f])
    if 'is_sae' in data:
        if data['is_sae'] and not report.report_deadline:
            report.report_deadline = datetime.utcnow() + timedelta(days=15)
        elif not data['is_sae']:
            report.report_deadline = None
    try:
        db.session.commit()
        return jsonify({'message': '수정됐어요', 'report': report.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/api/ae/<int:ae_id>/submit', methods=['POST'])
def ae_submit(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    report.is_submitted = True
    try:
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} 제출 완료 처리됐어요'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/api/ae/<int:ae_id>/delete', methods=['POST'])
def ae_delete(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    try:
        db.session.delete(report)
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} 삭제됐어요'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/api/ae/stats')
def ae_stats():
    reports = AEReport.query.all()
    if not reports: return jsonify({'message': '등록된 AE가 없어요'})
    grade_dist = {1:0,2:0,3:0,4:0,5:0}
    causality_dist, outcome_dist = {}, {}
    for r in reports:
        if r.ctcae_grade: grade_dist[r.ctcae_grade] = grade_dist.get(r.ctcae_grade,0) + 1
        if r.causality: causality_dist[r.causality] = causality_dist.get(r.causality,0) + 1
        if r.outcome: outcome_dist[r.outcome] = outcome_dist.get(r.outcome,0) + 1
    return jsonify({'total': len(reports), 'sae_count': sum(1 for r in reports if r.is_sae),
                    'submitted_count': sum(1 for r in reports if r.is_submitted),
                    'grade_distribution': grade_dist, 'causality_distribution': causality_dist,
                    'outcome_distribution': outcome_dist})

@main.route('/api/ae/<int:ae_id>/pdf')
def ae_pdf(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    title_style = ParagraphStyle('title', fontSize=16, spaceAfter=10, textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    header_style = ParagraphStyle('header', fontSize=12, spaceAfter=6, textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=10, spaceAfter=6, fontName='Helvetica')
    story = []
    story.append(Paragraph("Adverse Event Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(Spacer(1, 0.5*cm))
    if report.is_sae:
        story.append(Paragraph("⚠ SERIOUS ADVERSE EVENT (SAE)", ParagraphStyle('sae', fontSize=11, spaceAfter=8, textColor=colors.HexColor('#991b1b'), fontName='Helvetica-Bold')))
        story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Patient Information", header_style))
    story.append(_make_table([['Field','Value'],['Patient Code',report.patient_code],['Age',str(report.age) if report.age else 'N/A'],['Sex','Female' if report.sex=='F' else 'Male' if report.sex=='M' else 'N/A']]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Drug Information", header_style))
    story.append(_make_table([['Field','Value'],['Drug Name',report.drugname],['Dose',report.dose or 'N/A'],['Route',report.route or 'N/A']]))
    story.append(Spacer(1, 0.4*cm))
    grade_labels = {1:'Grade 1 (Mild)',2:'Grade 2 (Moderate)',3:'Grade 3 (Severe)',4:'Grade 4 (Life-threatening)',5:'Grade 5 (Death)'}
    story.append(Paragraph("Adverse Event Details", header_style))
    story.append(_make_table([['Field','Value'],['AE Term',report.ae_term],['CTCAE Grade',grade_labels.get(report.ctcae_grade,'N/A')],['SAE','YES' if report.is_sae else 'NO'],['Causality',report.causality or 'N/A'],['Outcome',report.outcome or 'N/A']]))
    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f'AE_{report.patient_code}_{report.id}.pdf', mimetype='application/pdf')

@main.route('/api/ae/<int:ae_id>/e2b')
def ae_e2b(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    sex_code = '1' if report.sex == 'M' else '2' if report.sex == 'F' else '0'
    outcome_map = {'회복':'1','회복중':'2','후유증 동반 회복':'3','미회복':'4','사망':'5','불명':'6'}
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
<ichicsr lang="en">
  <ichicsrmessageheader>
    <messagetype>ichicsr</messagetype>
    <messagenumb>AE-{report.id:06d}</messagenumb>
    <messagesenderidentifier>PHARMA-RISK-ANALYZER</messagesenderidentifier>
    <messagedate>{datetime.now().strftime('%Y%m%d%H%M%S')}</messagedate>
  </ichicsrmessageheader>
  <safetyreport>
    <safetyreportid>AE-{report.id:06d}</safetyreportid>
    <primarysourcecountry>KR</primarysourcecountry>
    <serious>{'1' if report.is_sae else '2'}</serious>
    {''.join(f'    <{k}>{v}</{k}>\n' for k,v in sae_flags.items())}
    <patient>
      <patientinitial>{report.patient_code}</patientinitial>
      <patientsex>{sex_code}</patientsex>
    </patient>
    <drug>
      <medicinalproduct>{report.drugname}</medicinalproduct>
    </drug>
    <reaction>
      <primarysourcereaction>{report.ae_term}</primarysourcereaction>
      <reactionoutcome>{outcome_code}</reactionoutcome>
    </reaction>
  </safetyreport>
</ichicsr>'''
    buf = io.BytesIO(xml.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f'E2B_AE-{report.id:06d}_{report.patient_code}.xml', mimetype='application/xml')

def _make_table(data):
    t = Table(data, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a56db')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f9fafb')]),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    return t
