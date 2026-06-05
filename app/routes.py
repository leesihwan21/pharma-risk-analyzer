from flask import Blueprint, render_template, jsonify, request, send_file  # send_file ???
import pandas as pd
import plotly.express as px
import plotly
import json
import os
import pycountry
import pickle
from ultralytics import YOLO
import cv2
import numpy as np
import networkx as nx
import threading
from PIL import Image
import io
import base64
import requests as http_requests
from flask import Response
from flask_mail import Message
from flask_restx import Api, Resource, fields, Namespace
from app import cache, limiter, mail
from app import cache, limiter
from app import cache
from datetime import datetime, timedelta, date
from flask_login import current_user
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from app.models import db, DrugSearch, FavoriteDrug, PredictionLog
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import db, DrugSearch, FavoriteDrug, PredictionLog, User, AEReport

main = Blueprint('main', __name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml')

def load_df():
    return pd.read_csv(DATA_PATH)

def load_model():
    model = pickle.load(open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb'))
    le_drug = pickle.load(open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'rb'))
    le_reac = pickle.load(open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'rb'))
    return model, le_drug, le_reac

def load_yolo():
    yolo_path = os.path.join(MODEL_DIR, 'best.pt')
    return YOLO(yolo_path)

def safe_encode(le, values):
    return [le.transform([v])[0] if v in le.classes_ else -1 for v in values]

@main.route('/api/detect', methods=['POST'])
@limiter.limit("20 per minute")
def detect_pill():
    if 'image' not in request.files:
        return jsonify({'error': '????? ?????? ??????'}), 400

    file = request.files['image']
    # ??????? ???? ????? ????? ???? ?켱 ????, ????? YOLO ????? ????
    drug_hint = request.form.get('drugname', '').upper()  
    sex = request.form.get('sex', 'F')
    age = float(request.form.get('age', 50))

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # 1. YOLO ???
    yolo = load_yolo()
    results = yolo(img)

    detections = []
    detected_drugs = []
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls].upper()  # ?빮??? ??????? ??????? ??? ????
            detections.append({
                'label': label,
                'confidence': round(conf * 100, 1)
            })
            detected_drugs.append(label)

    # ??? ??? ???? ????
    detected_drugs = list(set(detected_drugs))

    # ???? ????? ??? ????? ???? ?? Base64 ?????
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2. ??????? ?????/???? ???? ????
    risk_result = None
    combo_result = None
    
    # ?м??? ?????? ???? ???? ?? ???? (????? ?켱????, ?????? YOLO ???? ??)
    target_drug = drug_hint if drug_hint else (detected_drugs[0] if len(detected_drugs) > 0 else None)

    # [????? A] ???? ?? ????? ?? ???? ????
    if target_drug:
        try:
            model, le_drug, le_reac = load_model()
            risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
            
            if target_drug in le_drug.classes_:
                df = load_df()
                # FAERS ?????? ???? ???? ?????? ? ????? ???
                result_df = df[df['drugname'].str.upper() == target_drug]
                top_reac = result_df['pt'].value_counts().head(1)
                
                if len(top_reac) > 0:
                    reac = top_reac.index[0]
                    if reac in le_reac.classes_:
                        drug_enc = le_drug.transform([target_drug])[0]
                        reac_enc = le_reac.transform([reac])[0]
                        sex_enc = 0 if sex == 'F' else 1
                        
                        # ??????? ?????? ??o ????
                        drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
                        reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
                        combo_key = f"{drug_enc}_{reac_enc}"
                        combo_risk_rate = risk_rates['combo_risk'].get(combo_key, 0.5)

                        # ?????? ???? ?????? ?迭 ????
                        X = [[drug_enc, reac_enc, sex_enc, age, 
                              drug_risk_rate, reac_risk_rate, combo_risk_rate]]
                        
                        pred = model.predict(X)[0]
                        prob = model.predict_proba(X)[0]
                        
                        risk_result = {
                            'drug': target_drug,
                            'reaction': reac,
                            'risk_label': '?? ???? (???/???? ?????)' if pred == 1 else '? ??????',
                            'safe': round(float(prob[0]) * 100, 1),
                            'risk': round(float(prob[1]) * 100, 1)
                        }
                        
                        # DB ?α? ??????
                        log = PredictionLog(
                            drugname=target_drug, reaction=reac, age=age, sex=sex, risk=int(pred),
                            safe_prob=round(float(prob[0]) * 100, 1), risk_prob=round(float(prob[1]) * 100, 1)
                        )
                        db.session.add(log)
                        db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"???? ?? ???? ???? ????: {str(e)}")

    # [????? B] ???? ?? ????? 2?? ??? ????????? ?? ???? ????(Combo) ??? ????
    if len(detected_drugs) >= 2:
        try:
            model, le_drug, le_reac = load_model()
            risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
            sex_enc = 0 if sex == 'F' else 1
            combo_temp_results = []

            # ???? 2?? ???? ?м?
            for drug in detected_drugs[:2]:
                if drug not in le_drug.classes_:
                    continue
                drug_enc = le_drug.transform([drug])[0]
                drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)

                df = load_df()
                top_reacs = df[df['drugname'].str.upper() == drug]['pt'].value_counts().head(3).index.tolist()

                drug_results = []
                for reac in top_reacs:
                    if reac not in le_reac.classes_:
                        continue
                    reac_enc = le_reac.transform([reac])[0]
                    reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
                    combo_key = f"{drug_enc}_{reac_enc}"
                    combo_risk_rate = risk_rates['combo_risk'].get(combo_key, 0.5)

                    X_combo = [[drug_enc, reac_enc, sex_enc, age, 
                                drug_risk_rate, reac_risk_rate, combo_risk_rate]]
                    pred_c = model.predict(X_combo)[0]
                    prob_c = model.predict_proba(X_combo)[0]
                    
                    drug_results.append({
                        'reaction': reac,
                        'risk_label': '?? ????' if pred_c == 1 else '? ??????',
                        'risk_prob': round(float(prob_c[1]) * 100, 1)
                    })

                combo_temp_results.append({
                    'drug': drug,
                    'drug_risk_rate': round(drug_risk_rate * 100, 1),
                    'reactions': drug_results
                })
            
            if combo_temp_results:
                combo_result = combo_temp_results
        except Exception as e:
            print(f"???? ?м? ???? ????: {str(e)}")

    # 3. ???? ???? ???? ???
    return jsonify({
        'detections': detections,
        'image': img_b64,
        'risk_result': risk_result,
        'combo_result': combo_result  # ???????? ??????? 2?? ????? ?? ?????? ??????
    })

def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@cache.cached(timeout=300)
def dashboard():
    df = load_df()

    # ????? TOP 20
    top_reac = df['pt'].value_counts().head(20).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count',
                  title='Top 20 Adverse Reactions',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    # ?? TOP 20
    top_drug = df['drugname'].value_counts().head(20).reset_index()
    top_drug.columns = ['drug', 'count']
    fig2 = px.bar(top_drug, x='drug', y='count',
                  title='Top 20 Drugs by Report Count',
                  color='count', color_continuous_scale='Reds')
    fig2.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    # ???? ????
    sex_counts = df['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count',
                  title='Gender Distribution',
                  template='plotly_dark')
    fig3.update_layout(height=420)

    # ???? ??? ????
    outc_counts = df['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count',
                  title='Outcome Distribution',
                  color='count', color_continuous_scale='Greens')
    fig4.update_layout(template='plotly_dark', height=420)

    # ????뺰 ???? ???
    df['age_group'] = pd.cut(df['age'],
        bins=[0, 18, 30, 45, 60, 75, 120],
        labels=['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    )
    age_counts = df['age_group'].value_counts().sort_index().reset_index()
    age_counts.columns = ['age_group', 'count']
    fig5 = px.bar(age_counts, x='age_group', y='count',
                  title='????뺰 ????? ???? ???',
                  color='count', color_continuous_scale='Purples')
    fig5.update_layout(template='plotly_dark', height=420)

    # ?????? ????? ???? ???
    country_counts = df['reporter_country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts['iso3'] = country_counts['country'].apply(alpha2_to_alpha3)
    country_counts = country_counts.dropna(subset=['iso3'])

    fig6 = px.scatter_geo(
        country_counts,
        locations='iso3',
        size='count',
        hover_name='country',
        title='?????? ????? ???? ???',
        color='count',
        color_continuous_scale='Reds',
        projection='natural earth'
    )
    fig6.update_layout(
        template='plotly_dark',
        height=420,
        geo=dict(
        showframe=False,
        showcoastlines=True,
        showland=True,
        landcolor='#1e293b',
        bgcolor='#0f172a'
    )
)

    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        'chart4': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder),
        'chart5': json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder),
        'chart6': json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder),
    }

    return render_template('dashboard.html', charts=charts)

@main.route('/api/search/<drugname>')
@cache.cached(timeout=300)
def search_drug(drugname):
    df = load_df()
    result = df[df['drugname'].str.upper() == drugname.upper()]

    if len(result) == 0:
        return jsonify({'error': f'???? a?? ?? ??????: {drugname.upper()}'}), 404

    top_reac = result['pt'].value_counts().head(10).reset_index()
    top_reac.columns = ['reaction', 'count']
    age_data = result['age'].dropna()
    age_avg = round(float(age_data.mean()), 1) if len(age_data) > 0 else 0
    sex_counts = result['sex'].value_counts().to_dict()
    outc_counts = result['outc_cod'].value_counts().to_dict()

    try:
        log = DrugSearch(
            drugname=drugname.upper(),
            total_reports=len(result),
            age_avg=age_avg
    )
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

@main.route('/api/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    data = request.get_json()
    drugname = data.get('drugname', '').upper()
    reaction = data.get('reaction', '').upper()
    sex = data.get('sex', 'F')
    age = float(data.get('age', 50))

    model, le_drug, le_reac = load_model()

    if drugname not in le_drug.classes_:
        return jsonify({'error': f'?? ?? ???? ??: {drugname}'}), 400
    if reaction not in le_reac.classes_:
        return jsonify({'error': f'?? ?? ???? ?????: {reaction}'}), 400

    drug_enc = le_drug.transform([drugname])[0]
    reac_enc = le_reac.transform([reaction])[0]
    sex_enc = 0 if sex == 'F' else 1

    # ??? ?????? ?????? ?ε?
    risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
    drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
    reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
    combo_key = f"{drug_enc}_{reac_enc}"
    combo_risk_rate = risk_rates['combo_risk'].get(combo_key, 0.5)

    X = [[drug_enc, reac_enc, sex_enc, age,
          drug_risk_rate, reac_risk_rate, combo_risk_rate]]

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    try:
        log = PredictionLog(
            drugname=drugname.upper(),
            reaction=reaction.upper(),
            age=age,
            sex=sex,
            risk=int(pred),
            safe_prob=round(float(prob[0]) * 100, 1),
            risk_prob=round(float(prob[1]) * 100, 1)
        )
        db.session.add(log)
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({
        'drug': drugname,
        'reaction': reaction,
        'risk': int(pred),
        'risk_label': '?? ???? (???/???? ?????)' if pred == 1 else '? ??????',
        'probability': {
            'safe': round(float(prob[0]) * 100, 1),
            'risk': round(float(prob[1]) * 100, 1)
        }
    })

@main.route('/api/combo', methods=['POST'])
def combo_risk():
    data = request.get_json()
    drug1 = data.get('drug1', '').upper()
    drug2 = data.get('drug2', '').upper()
    age = float(data.get('age', 50))
    sex = data.get('sex', 'F')

    model, le_drug, le_reac = load_model()
    risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))

    if drug1 not in le_drug.classes_:
        return jsonify({'error': f'?? ?? ???? ??: {drug1}'}), 400
    if drug2 not in le_drug.classes_:
        return jsonify({'error': f'?? ?? ???? ??: {drug2}'}), 400

    sex_enc = 0 if sex == 'F' else 1
    results = []

    for drug in [drug1, drug2]:
        drug_enc = le_drug.transform([drug])[0]
        drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)

        # ??? ???? ???? ???? ????? top 5
        df = load_df()
        top_reacs = df[df['drugname'].str.upper() == drug]['pt'].value_counts().head(5).index.tolist()

        drug_results = []
        for reac in top_reacs:
            if reac not in le_reac.classes_:
                continue
            reac_enc = le_reac.transform([reac])[0]
            reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
            combo_key = f"{drug_enc}_{reac_enc}"
            combo_risk_rate = risk_rates['combo_risk'].get(combo_key, 0.5)

            X = [[drug_enc, reac_enc, sex_enc, age,
                  drug_risk_rate, reac_risk_rate, combo_risk_rate]]
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            drug_results.append({
                'reaction': reac,
                'risk': int(pred),
                'risk_label': '?? ????' if pred == 1 else '? ??????',
                'risk_prob': round(float(prob[1]) * 100, 1)
            })

        results.append({
            'drug': drug,
            'drug_risk_rate': round(drug_risk_rate * 100, 1),
            'reactions': drug_results
        })

    return jsonify({'results': results})

@main.route('/api/llm_explain', methods=['POST'])
@limiter.limit("10 per minute")
def llm_explain():
    data = request.get_json()
    drug = data.get('drug', '')
    reaction = data.get('reaction', '')
    risk_label = data.get('risk_label', '')
    safe_prob = data.get('safe_prob', 0)
    risk_prob = data.get('risk_prob', 0)

    prompt = f"""?? ????? ?м? ?????? ?????? ??????????? (3-4????, ???????):
??: {drug}
?????: {reaction}
AI ????: {risk_label}
???? ???: {safe_prob}% / ???? ???: {risk_prob}%

?????? ???????? ?? ?????? ???? ????????? ???????????."""

    try:
        response = http_requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )
        result = response.json()
        return jsonify({'explanation': result['response']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/api/autocomplete/<keyword>')
@cache.cached(timeout=600)
def autocomplete(keyword):
    df = load_df()
    keyword = keyword.upper()
    drugs = df['drugname'].str.upper().unique()
    matches = [d for d in drugs if d.startswith(keyword)][:10]
    return jsonify({'suggestions': sorted(matches)})


@main.route('/drug/<drugname>')
@cache.cached(timeout=300)
def drug_detail(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return "???? a?? ?? ??????", 404

    # ????? TOP 15
    top_reac = result['pt'].value_counts().head(15).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count',
                  title=f'{drugname} - ????? TOP 15',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    # ???? ????
    age_data = result['age'].dropna()
    fig2 = px.histogram(age_data, nbins=20,
                        title=f'{drugname} - ??? ???? ????',
                        color_discrete_sequence=['#38bdf8'])
    fig2.update_layout(template='plotly_dark', height=420)

    # ???? ????
    sex_counts = result['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count',
                  title=f'{drugname} - ???? ????',
                  template='plotly_dark')
    fig3.update_layout(height=420)

    # ???? ????
    outc_counts = result['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count',
                  title=f'{drugname} - ???? ????',
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


@main.route('/api/report/<drugname>')
def generate_report(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '???? a?? ?? ??????'}), 404

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

    # ????
    story.append(Paragraph(f"Pharma Risk Analyzer - Drug Report", title_style))
    story.append(Paragraph(f"Drug: {drugname}", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(Spacer(1, 0.5*cm))

    # ?? ????
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

    # Top 10 ?????
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

    # ???? ????
    story.append(Paragraph("Outcome Distribution", header_style))
    outc_data = [['Outcome Code', 'Description', 'Count']]
    outc_map = {'DE': 'Death', 'HO': 'Hospitalization', 'LT': 'Life Threatening',
                'DS': 'Disability', 'CA': 'Congenital Anomaly', 'OT': 'Other', 'RI': 'Required Intervention'}
    for code, cnt in result['outc_cod'].value_counts().items():
        outc_data.append([str(code), outc_map.get(str(code), str(code)), str(cnt)])
    t3 = Table(outc_data, colWidths=[4*cm, 9*cm, 4*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a56db')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t3)

    doc.build(story)
    buf.seek(0)

    return send_file(buf, as_attachment=True,
                     download_name=f'{drugname}_report.pdf',
                     mimetype='application/pdf')


@main.route('/api/favorite/<drugname>', methods=['POST'])
def add_favorite(drugname):
    try:
        fav = FavoriteDrug(drugname=drugname.upper())
        db.session.add(fav)
        db.session.commit()
        return jsonify({'message': f'{drugname} ????a?? ???!'})
    except:
        db.session.rollback()
        return jsonify({'message': '??? ????a?? ?????'})

@main.route('/api/favorites')
def get_favorites():
    favs = FavoriteDrug.query.order_by(FavoriteDrug.added_at.desc()).all()
    return jsonify({'favorites': [f.to_dict() for f in favs]})

@main.route('/api/history')
def get_history():
    searches = DrugSearch.query.order_by(DrugSearch.searched_at.desc()).limit(20).all()
    predictions = PredictionLog.query.order_by(PredictionLog.predicted_at.desc()).limit(20).all()
    return jsonify({
        'searches': [s.to_dict() for s in searches],
        'predictions': [p.to_dict() for p in predictions]
    })


@main.route('/api/network/<drugname>')
@cache.cached(timeout=300)
def drug_network(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '???? a?? ?? ??????'}), 404

    # ???? 15?? ?????
    top_reac = result['pt'].value_counts().head(15)

    nodes = [{'id': drugname, 'type': 'drug', 'size': 30}]
    edges = []

    for reac, cnt in top_reac.items():
        nodes.append({'id': reac, 'type': 'reaction', 'size': max(8, min(20, cnt // 10))})
        edges.append({'source': drugname, 'target': reac, 'weight': int(cnt)})

    return jsonify({'nodes': nodes, 'edges': edges})


# ???? ????
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    yolo = load_yolo()
    while True:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
        if not success:
            break

        # YOLOv8 ???
        results = yolo(frame, verbose=False)
        frame = results[0].plot()

        # JPEG ?????
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@main.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/webcam')
def webcam():
    return render_template('webcam.html')

@main.route('/api/stop_camera')
def stop_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
    return jsonify({'message': '???? ??????'})


@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        email = data.get('email', '')
        password = data.get('password', '')

        if User.query.filter_by(username=username).first():
            return jsonify({'error': '??? ???? ???? ???????'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': '??? ???? ???? ??????????'}), 400

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return jsonify({'message': '??????? ????!', 'username': username})

    return render_template('register.html')


@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': '????? ??? ???й???? ?????'}), 401

        login_user(user)
        return jsonify({'message': f'{username}?? ??????!', 'username': username})

    return render_template('login.html')


@main.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': '?α????????'})


@main.route('/api/me')
def me():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

@main.route('/api/send_report/<drugname>', methods=['POST'])
@login_required
def send_report_email(drugname):
    data = request.get_json()
    email = data.get('email', '')
    if not email:
        return jsonify({'error': '??????? ??????????'}), 400

    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '???? a?? ?? ??????'}), 404

    top_reac = result['pt'].value_counts().head(5)
    age_data = result['age'].dropna()

    # ????? ????
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
        body += f"{i}. {reac}: {cnt}??\n"

    body += "\n\nPowered by Pharma Risk Analyzer"

    try:
        msg = Message(
            subject=f'[Pharma Risk Analyzer] {drugname} ?м? ?????',
            recipients=[email],
            body=body
        )
        mail.send(msg)
        return jsonify({'message': f'{email} ?? ??????? ???????!'})
    except Exception as e:
        return jsonify({'error': f'???? ??? ????: {str(e)}'}), 500
    
@main.route('/api/timeline/<drugname>')
@cache.cached(timeout=300)
def drug_timeline(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': '???? a?? ?? ??????'}), 404

    # ?????? ???? ???
    result = result.copy()
    result['year'] = pd.to_datetime(result['event_dt'], errors='coerce').dt.year
    yearly = result.groupby('year').size().reset_index(name='count')
    yearly = yearly.dropna(subset=['year'])
    yearly['year'] = yearly['year'].astype(int)
    yearly = yearly[yearly['year'] >= 2000]

    return jsonify({
        'drugname': drugname,
        'timeline': yearly.to_dict(orient='records')
    })

@main.route('/compare')
def compare():
    return render_template('compare.html')

@main.route('/api/compare')
def compare_drugs():
    drug1 = request.args.get('drug1', '').upper()
    drug2 = request.args.get('drug2', '').upper()

    if not drug1 or not drug2:
        return jsonify({'error': '?? 2???? ??????????'}), 400

    df = load_df()
    result1 = df[df['drugname'].str.upper() == drug1]
    result2 = df[df['drugname'].str.upper() == drug2]

    if len(result1) == 0:
        return jsonify({'error': f'{drug1} ?? a?? ?? ??????'}), 404
    if len(result2) == 0:
        return jsonify({'error': f'{drug2} ?? a?? ?? ??????'}), 404

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

    return jsonify({
        'drug1': get_stats(result1, drug1),
        'drug2': get_stats(result2, drug2)
    })

@main.route('/api/filter')
def filter_data():
    drug = request.args.get('drug', '').upper()
    sex = request.args.get('sex', '')
    age_min = request.args.get('age_min', 0, type=int)
    age_max = request.args.get('age_max', 120, type=int)
    outcome = request.args.get('outcome', '')
    country = request.args.get('country', '')

    df = load_df()

    if drug:
        df = df[df['drugname'].str.upper() == drug]
    if sex:
        df = df[df['sex'] == sex]
    if age_min:
        df = df[df['age'] >= age_min]
    if age_max < 120:
        df = df[df['age'] <= age_max]
    if outcome:
        df = df[df['outc_cod'] == outcome]
    if country:
        df = df[df['reporter_country'].str.upper() == country.upper()]

    if len(df) == 0:
        return jsonify({'error': '????? ?´? ??????? ??????'}), 404

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

@main.route('/filter')
def filter_page():
    return render_template('filter.html')


KOREA_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'raw', 'korea_adr.csv')

@main.route('/korea')
def korea_dashboard():
    df = pd.read_csv(KOREA_DATA_PATH, encoding='cp949')

    # 2024?? Top 10 ????
    fig1 = px.bar(
        df.head(10),
        x='??????????(2024)',
        y='?????????????(2024)',
        title='??? 2024?? Top 10 ???????',
        color='?????????????(2024)',
        color_continuous_scale='Blues'
    )
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    # ?????? ????? (Top 5 ????)
    years = ['2019', '2020', '2021', '2022', '2023', '2024']
    top5 = df.head(5)['??????????(2024)'].tolist()

    traces = []
    for symptom in top5:
        row = df[df['??????????(2024)'] == symptom]
        if len(row) == 0:
            continue
        counts = []
        for y in years:
            col = f'?????????????({y})'
            if col in df.columns:
                counts.append(int(row[col].values[0]) if len(row[col].values) > 0 else 0)
            else:
                counts.append(0)
        traces.append({'x': years, 'y': counts, 'name': symptom, 'type': 'scatter', 'mode': 'lines+markers'})

    fig2 = px.line(title='??? Top 5 ???? ?????? ?????')
    for t in traces:
        fig2.add_scatter(x=t['x'], y=t['y'], name=t['name'], mode='lines+markers')
    fig2.update_layout(template='plotly_dark', height=420)

    # 2024 vs 2023 ????
    fig3 = px.bar(
        df.head(10),
        x='??????????(2024)',
        y=['?????????????(2024)', '?????????????(2023)'],
        title='2024 vs 2023 Top 10 ???? ????',
        barmode='group',
        color_discrete_sequence=['#38bdf8', '#a78bfa']
    )
    fig3.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
    }

    return render_template('korea.html', charts=charts)

# ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??? ??? routes.py ?? ????? ?????? ???????
# ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

# CTCAE Grade ??? ???? ????? ????
CTCAE_KEYWORDS = {
    5: ['death', 'fatal', '????'],
    4: ['life-threatening', 'life threatening', '????????', 'icu', 'ventilat'],
    3: ['hospitali', 'severe', '???', '????', 'severe'],
    2: ['moderate', '???', 'limiting'],
    1: ['mild', '????', 'minor'],
}
 
SAE_KEYWORDS = ['????', '???', '????????', '????????', '??o??', 'death', 'hospitali',
                'life-threatening', 'disability', 'congenital']
 
 
def auto_ctcae_grade(ae_term: str) -> int:
    """AE ?????? CTCAE Grade ??? ???? (1~5)"""
    term_lower = ae_term.lower()
    for grade in [5, 4, 3, 2, 1]:
        for kw in CTCAE_KEYWORDS[grade]:
            if kw in term_lower:
                return grade
    return 1  # ???? Grade 1
 
 
def auto_is_sae(ae_term: str, ctcae_grade: int) -> bool:
    """SAE ??? ????"""
    if ctcae_grade >= 3:
        return True
    term_lower = ae_term.lower()
    return any(kw in term_lower for kw in SAE_KEYWORDS)
 
 
# ???? AE ?????? ????????????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/ae_manager')
def ae_manager():
    return render_template('ae_manager.html')
 
 
# ???? AE ???? ??? ??????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/list')
def ae_list():
    status_filter = request.args.get('status', '')   # overdue / urgent / warning / normal / submitted
    sae_only = request.args.get('sae_only', 'false') == 'true'
 
    query = AEReport.query.order_by(AEReport.reported_at.desc())
 
    if sae_only:
        query = query.filter_by(is_sae=True)
 
    reports = query.all()
 
    # status ????? Python ???????? o?? (deadline_status?? ???? ???????????)
    result = [r.to_dict() for r in reports]
    if status_filter:
        result = [r for r in result if r['deadline_status'] == status_filter]
 
    # ???? ????
    all_reports = [r.to_dict() for r in AEReport.query.all()]
    summary = {
        'total': len(all_reports),
        'sae_count': sum(1 for r in all_reports if r['is_sae']),
        'overdue': sum(1 for r in all_reports if r['deadline_status'] == 'overdue'),
        'urgent': sum(1 for r in all_reports if r['deadline_status'] == 'urgent'),
        'submitted': sum(1 for r in all_reports if r['is_submitted']),
    }
 
    return jsonify({'reports': result, 'summary': summary})
 
 
# ???? AE ??? ??? ??????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/<int:ae_id>')
def ae_detail(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    return jsonify(report.to_dict())
 
 
# ???? AE ???? ????????????????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/create', methods=['POST'])
def ae_create():
    data = request.get_json()
 
    # ????? u?
    if not data.get('patient_code') or not data.get('drugname') or not data.get('ae_term'):
        return jsonify({'error': '??????, ????, AE ?????? ???????'}), 400
 
    ae_term = data.get('ae_term', '')
 
    # CTCAE Grade ??? ???? (???? ????? ?????? ?켱)
    ctcae_grade = int(data.get('ctcae_grade') or auto_ctcae_grade(ae_term))
 
    # SAE ??? ???? (???? u? ?켱)
    is_sae_input = data.get('is_sae')
    if is_sae_input is not None:
        is_sae = bool(is_sae_input)
    else:
        is_sae = auto_is_sae(ae_term, ctcae_grade)
 
    # ???? ?????? ???? (SAE?? 15??, ??? AE?? null)
    report_deadline = None
    if is_sae:
        report_deadline = datetime.utcnow() + timedelta(days=15)
 
    # ??￥ ???
    ae_start = None
    ae_end = None
    try:
        if data.get('ae_start_date'):
            ae_start = datetime.strptime(data['ae_start_date'], '%Y-%m-%d').date()
        if data.get('ae_end_date'):
            ae_end = datetime.strptime(data['ae_end_date'], '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': '??￥ ?????? ?u???? ???? (YYYY-MM-DD)'}), 400
 
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
        return jsonify({
            'message': 'AE ???????? ????????',
            'id': report.id,
            'is_sae': is_sae,
            'ctcae_grade': ctcae_grade,
            'report_deadline': report_deadline.strftime('%Y-%m-%d') if report_deadline else None
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
 
 
# ???? AE ???? ????????????????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/<int:ae_id>/update', methods=['POST'])
def ae_update(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    data = request.get_json()
 
    # ???? ?????? ??常 ???????
    fields = ['ae_term', 'ctcae_grade', 'is_sae', 'sae_category',
              'causality', 'action_taken', 'outcome', 'notes',
              'dose', 'route', 'age', 'sex']
    for f in fields:
        if f in data:
            setattr(report, f, data[f])
 
    # SAE ???? ?? ?????? ??????
    if 'is_sae' in data:
        if data['is_sae'] and not report.report_deadline:
            report.report_deadline = datetime.utcnow() + timedelta(days=15)
        elif not data['is_sae']:
            report.report_deadline = None
 
    try:
        db.session.commit()
        return jsonify({'message': '?????????', 'report': report.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
 
 
# ???? AE ???? ??? o?? ????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/<int:ae_id>/submit', methods=['POST'])
def ae_submit(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    report.is_submitted = True
    try:
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} ???? ??? o???????'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
 
 
# ???? AE ???? ????????????????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/<int:ae_id>/delete', methods=['POST'])
def ae_delete(ae_id):
    report = AEReport.query.get_or_404(ae_id)
    try:
        db.session.delete(report)
        db.session.commit()
        return jsonify({'message': f'AE #{ae_id} ?????????'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
 
 
# ???? AE ???? ???? ??????????????????????????????????????????????????????????????????????????????????????????????
 
@main.route('/api/ae/stats')
def ae_stats():
    reports = AEReport.query.all()
    if not reports:
        return jsonify({'message': '????? AE?? ??????'})
 
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

@main.route('/api/ae/<int:ae_id>/pdf')
def ae_pdf(ae_id):
    report = AEReport.query.get_or_404(ae_id)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', fontSize=16, spaceAfter=10,
                                  textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    header_style = ParagraphStyle('header', fontSize=12, spaceAfter=6,
                                   textColor=colors.HexColor('#1a56db'), fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=10, spaceAfter=6, fontName='Helvetica')

    story = []

    # ????
    story.append(Paragraph("Adverse Event Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(Spacer(1, 0.5*cm))

    # SAE ???? ???
    if report.is_sae:
        sae_style = ParagraphStyle('sae', fontSize=11, spaceAfter=8,
                                    textColor=colors.HexColor('#991b1b'),
                                    fontName='Helvetica-Bold')
        story.append(Paragraph("? SERIOUS ADVERSE EVENT (SAE)", sae_style))
        if report.report_deadline:
            story.append(Paragraph(
                f"Reporting Deadline: {report.report_deadline.strftime('%Y-%m-%d')} "
                f"({report.days_until_deadline()}days remaining)",
                ParagraphStyle('deadline', fontSize=10, textColor=colors.HexColor('#991b1b'), fontName='Helvetica')
            ))
        story.append(Spacer(1, 0.3*cm))

    # ??? ????
    story.append(Paragraph("Patient Information", header_style))
    patient_data = [
        ['Field', 'Value'],
        ['Patient Code', report.patient_code],
        ['Age', str(report.age) if report.age else 'N/A'],
        ['Sex', 'Female' if report.sex == 'F' else 'Male' if report.sex == 'M' else 'N/A'],
    ]
    t = _make_table(patient_data)
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ?? ????
    story.append(Paragraph("Drug Information", header_style))
    drug_data = [
        ['Field', 'Value'],
        ['Drug Name', report.drugname],
        ['Dose', report.dose or 'N/A'],
        ['Route', report.route or 'N/A'],
    ]
    t2 = _make_table(drug_data)
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # AE ????
    story.append(Paragraph("Adverse Event Details", header_style))
    grade_labels = {1:'Grade 1 (Mild)', 2:'Grade 2 (Moderate)', 3:'Grade 3 (Severe)',
                    4:'Grade 4 (Life-threatening)', 5:'Grade 5 (Death)'}
    ae_data = [
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
    ]
    t3 = _make_table(ae_data)
    story.append(t3)
    story.append(Spacer(1, 0.4*cm))

    # ???? ????
    story.append(Paragraph("Reporting Information", header_style))
    report_data = [
        ['Field', 'Value'],
        ['Report Date', report.reported_at.strftime('%Y-%m-%d %H:%M')],
        ['Deadline', report.report_deadline.strftime('%Y-%m-%d') if report.report_deadline else 'N/A'],
        ['Status', 'Submitted' if report.is_submitted else 'Pending'],
    ]
    t4 = _make_table(report_data)
    story.append(t4)

    if report.notes:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("Notes", header_style))
        story.append(Paragraph(report.notes, sub_style))

    doc.build(story)
    buf.seek(0)

    return send_file(buf, as_attachment=True,
                     download_name=f'AE_{report.patient_code}_{report.id}.pdf',
                     mimetype='application/pdf')


def _make_table(data):
    """PDF ????? ???? ????"""
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

# ???? ICH E2B XML ???? ????????????????????????????????????????????????????????????????????????????????????

@main.route('/api/ae/<int:ae_id>/e2b')
def ae_e2b(ae_id):
    report = AEReport.query.get_or_404(ae_id)

    # ???? ??? (ICH E2B: 1=????, 2=????, 0=???)
    sex_code = '1' if report.sex == 'M' else '2' if report.sex == 'F' else '0'

    # ???? ??? (ICH E2B)
    outcome_map = {
        '???': '1', '?????': '2', '?????? ???? ???': '3',
        '?????': '4', '????': '5', '???': '6'
    }
    outcome_code = outcome_map.get(report.outcome or '', '6')

    # SAE ?з? ???
    sae_flags = {
        'seriousnessother': '1',
        'seriousnesshospitalization': '1' if report.sae_category == '???' else '0',
        'seriousnesslifethreatening': '1' if report.sae_category == '????????' else '0',
        'seriousnessdisabling': '1' if report.sae_category == '????????' else '0',
        'seriousnesscongenitalanomali': '1' if report.sae_category == '??o?????' else '0',
        'seriousnessdeath': '1' if report.sae_category == '????' else '0',
    }

    # ??????? ??? (1=????, 2=????)
    causality_code = '1' if report.causality in ['Certain','Probable','Possible'] else '2'

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
    <receiptdateformat>102</receiptdateformat>
    <receiptdate>{report.reported_at.strftime('%Y%m%d')}</receiptdate>

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
      <actiondrug>{'1' if report.action_taken == '???????' else '2' if report.action_taken == '?뷮????' else '3'}</actiondrug>
    </drug>

    <reaction>
      <primarysourcereaction>{report.ae_term}</primarysourcereaction>
      <reactionmeddrapt>{report.ae_term}</reactionmeddrapt>
      {'<reactionstartdateformat>102</reactionstartdateformat>' if report.ae_start_date else ''}
      {'<reactionstartdate>' + report.ae_start_date.strftime('%Y%m%d') + '</reactionstartdate>' if report.ae_start_date else ''}
      {'<reactionenddateformat>102</reactionenddateformat>' if report.ae_end_date else ''}
      {'<reactionenddate>' + report.ae_end_date.strftime('%Y%m%d') + '</reactionenddate>' if report.ae_end_date else ''}
      <reactionoutcome>{outcome_code}</reactionoutcome>
    </reaction>

    <summary>
      <narrativeincludeclinical>
        Patient: {report.patient_code} | Drug: {report.drugname} | Reaction: {report.ae_term}
        CTCAE Grade: {report.ctcae_grade} | SAE: {'Yes' if report.is_sae else 'No'}
        Causality: {report.causality or 'Unknown'} | Outcome: {report.outcome or 'Unknown'}
        {'Reporting Deadline: ' + report.report_deadline.strftime('%Y-%m-%d') if report.report_deadline else ''}
      </narrativeincludeclinical>
    </summary>

  </safetyreport>
</ichicsr>'''

    buf = io.BytesIO(xml.encode('utf-8'))
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name=f'E2B_AE-{report.id:06d}_{report.patient_code}.xml',
        mimetype='application/xml'
    )

# ???? PRR (Proportional Reporting Ratio) ??? ??? ??????????????????????????
 
@main.route('/api/prr/<drugname>')
@cache.cached(timeout=600)
def calculate_prr(drugname):
    """
    PRR (Proportional Reporting Ratio) ????
    FDA/EMA???? ??????? ?? ????? ??? ??? ???
 
    PRR = (a/b) / (c/d)
    a = ??A + ?????X ???? ???
    b = ??A ??u ???? ???
    c = ??? ?? + ?????X ???? ???
    d = ??? ?? ??u ???? ???
 
    ??? ????: PRR >= 2 AND ??????? >= 3
    """
    df = load_df()
    drugname = drugname.upper()
 
    drug_reports = df[df['drugname'].str.upper() == drugname]
    if len(drug_reports) == 0:
        return jsonify({'error': f'???? a?? ?? ??????: {drugname}'}), 404
 
    other_reports = df[df['drugname'].str.upper() != drugname]
 
    total_drug = len(drug_reports)
    total_other = len(other_reports)
 
    if total_other == 0:
        return jsonify({'error': '???? ??????? ???????'}), 400
 
    # ???? Top 20 ????뿡 ???? PRR ????
    top_reactions = drug_reports['pt'].value_counts().head(20).index.tolist()
 
    results = []
    for reac in top_reactions:
        a = len(drug_reports[drug_reports['pt'] == reac])
        b = total_drug
        c = len(other_reports[other_reports['pt'] == reac])
        d = total_other
 
        if c == 0 or b == 0:
            continue
 
        prr = (a / b) / (c / d)
        
        # 95% ?????? ???? (?α? PRR ????)
        import math
        try:
            se = math.sqrt((1/a) - (1/b) + (1/c) - (1/d))
            prr_lower = math.exp(math.log(prr) - 1.96 * se)
            prr_upper = math.exp(math.log(prr) + 1.96 * se)
        except (ValueError, ZeroDivisionError):
            prr_lower = prr_upper = prr
 
        # ??? ???? (Evans ????: PRR >= 2, n >= 3, Chi-square >= 4)
        is_signal = prr >= 2 and a >= 3
 
        results.append({
            'reaction': reac,
            'drug_count': int(a),           # ??A???? ??? ????? ???
            'drug_total': int(b),           # ??A ??u ???
            'other_count': int(c),          # ??? ?????? ??? ????? ???
            'other_total': int(d),          # ??? ?? ??u ???
            'drug_pct': round(a/b*100, 2),  # ??A???? ????
            'other_pct': round(c/d*100, 2), # ??? ?????? ????
            'prr': round(prr, 2),
            'prr_lower': round(prr_lower, 2),
            'prr_upper': round(prr_upper, 2),
            'is_signal': is_signal,
            'signal_level': (
                '?? ???? ???' if prr >= 5 and a >= 3 else
                '?? ???' if prr >= 2 and a >= 3 else
                '? ?????'
            )
        })
 
    # PRR ???? ?????? ????
    results.sort(key=lambda x: x['prr'], reverse=True)
 
    # ??? ????
    signal_count = sum(1 for r in results if r['is_signal'])
    strong_signal_count = sum(1 for r in results if r['prr'] >= 5 and r['drug_count'] >= 3)
 
    return jsonify({
        'drugname': drugname,
        'total_reports': total_drug,
        'signal_count': signal_count,
        'strong_signal_count': strong_signal_count,
        'results': results
    })
 
 
@main.route('/prr')
def prr_page():
    return render_template('prr.html')



@main.route('/trend')
def trend_page():
    return render_template('trend.html')


@main.route('/api/trend')
def api_trend():
    import pandas as pd
    from flask import jsonify, request
    df = pd.read_csv('data/processed/processed_faers.csv')
    drugname = request.args.get('drug', '').upper()
    if not drugname:
        return jsonify({'error': 'no drug'}), 400
    filtered = df[df['drugname'] == drugname]
    if filtered.empty:
        return jsonify({'error': 'not found'}), 404
    trend = filtered.groupby('quarter').size().reset_index(name='count')
    trend = trend.sort_values('quarter')
    return jsonify({'drug': drugname, 'trend': trend.to_dict(orient='records')})
@main.route('/shap')
def shap_page():
    return render_template('shap.html')


@main.route('/api/shap')
def api_shap():
    import pickle
    import shap
    import numpy as np
    import pandas as pd
    from flask import jsonify, request

    drugname = request.args.get('drug', '').upper()
    reaction = request.args.get('reaction', '').upper()
    age = float(request.args.get('age', 50))
    sex = request.args.get('sex', 'F')

    try:
        model = pickle.load(open('ml/model.pkl', 'rb'))
        le_drug = pickle.load(open('ml/le_drug.pkl', 'rb'))
        le_reac = pickle.load(open('ml/le_reac.pkl', 'rb'))
        risk_rates = pickle.load(open('ml/risk_rates.pkl', 'rb'))
    except Exception as e:
        return jsonify({'error': 'model load failed: ' + str(e)}), 500

    if drugname not in le_drug.classes_:
        return jsonify({'error': 'unknown drug: ' + drugname}), 400
    if reaction not in le_reac.classes_:
        return jsonify({'error': 'unknown reaction: ' + reaction}), 400

    drug_enc = le_drug.transform([drugname])[0]
    reac_enc = le_reac.transform([reaction])[0]
    sex_enc = 0 if sex == 'F' else 1
    drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
    reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
    combo_risk_rate = risk_rates['combo_risk'].get(f"{drug_enc}_{reac_enc}", 0.5)

    feature_names = ['drug', 'reaction', 'sex', 'age', 'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']
    X = np.array([[drug_enc, reac_enc, sex_enc, age, drug_risk_rate, reac_risk_rate, combo_risk_rate]])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]

    feature_display = {
        'drug': drugname,
        'reaction': reaction,
        'sex': sex,
        'age': age,
        'drug_risk_rate': round(drug_risk_rate, 3),
        'reac_risk_rate': round(reac_risk_rate, 3),
        'combo_risk_rate': round(combo_risk_rate, 3)
    }

    shap_result = []
    for i, name in enumerate(feature_names):
        shap_result.append({
            'feature': name,
            'value': feature_display[name],
            'shap': round(float(sv[i]), 4)
        })

    shap_result.sort(key=lambda x: abs(x['shap']), reverse=True)

    return jsonify({
        'drug': drugname,
        'reaction': reaction,
        'prediction': pred,
        'risk_label': 'HIGH RISK' if pred == 1 else 'LOW RISK',
        'probability': {
            'safe': round(float(prob[0]) * 100, 1),
            'risk': round(float(prob[1]) * 100, 1)
        },
        'shap': shap_result
    })

@main.route('/drug-lookup')
def drug_lookup_page():
    return render_template('drug_lookup.html')
 
 
@main.route('/api/drug-lookup')
def api_drug_lookup():
    import requests as req
    from flask import jsonify, request, current_app
 
    drugname = request.args.get('name', '').strip()
    if not drugname:
        return jsonify({'error': 'no name'}), 400
 
    api_key = current_app.config.get('MFDS_API_KEY', '')
    results = {'korean': None, 'openfda': None}
 
    # 1. ???o ?????? API (??? ??)
    try:
        mfds_url = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'
        params = {
            'serviceKey': api_key,
            'item_name': drugname,
            'type': 'json',
            'numOfRows': 1
        }
        r = req.get(mfds_url, params=params, timeout=5)
        data = r.json()
        items = data.get('body', {}).get('items', [])
        if items:
            item = items[0]
            results['korean'] = {
                'name': item.get('ITEM_NAME', ''),
                'company': item.get('ENTP_NAME', ''),
                'shape': item.get('DRUG_SHAPE', ''),
                'color': item.get('COLOR_CLASS1', ''),
                'etc_otc': item.get('ETC_OTC_CODE', ''),
                'class_name': item.get('CLASS_NAME', ''),
                'img_url': item.get('ITEM_IMAGE', '')
            }
    except Exception as e:
        results['korean_error'] = str(e)
 
    # 2. OpenFDA Drug Label API (??? ??)
    try:
        fda_url = 'https://api.fda.gov/drug/label.json'
        params = {
            'search': f'openfda.brand_name:"{drugname.upper()}" OR openfda.generic_name:"{drugname.upper()}"',
            'limit': 1
        }
        r = req.get(fda_url, params=params, timeout=5)
        data = r.json()
        items = data.get('results', [])
        if items:
            item = items[0]
            openfda = item.get('openfda', {})
            results['openfda'] = {
                'brand_name': openfda.get('brand_name', [''])[0] if openfda.get('brand_name') else '',
                'generic_name': openfda.get('generic_name', [''])[0] if openfda.get('generic_name') else '',
                'manufacturer': openfda.get('manufacturer_name', [''])[0] if openfda.get('manufacturer_name') else '',
                'purpose': item.get('purpose', [''])[0][:300] if item.get('purpose') else '',
                'warnings': item.get('warnings', [''])[0][:300] if item.get('warnings') else '',
                'adverse_reactions': item.get('adverse_reactions', [''])[0][:500] if item.get('adverse_reactions') else '',
                'dosage': item.get('dosage_and_administration', [''])[0][:300] if item.get('dosage_and_administration') else ''
            }
    except Exception as e:
        results['openfda_error'] = str(e)
 
    if not results['korean'] and not results['openfda']:
        return jsonify({'error': 'not found', 'drug': drugname}), 404
 
    results['drug'] = drugname
    return jsonify(results)

@main.route('/api/drug-vision', methods=['POST'])
def api_drug_vision():
    import base64
    import json
    import requests as req
    from flask import jsonify, request, current_app
 
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400
 
    image_file = request.files['image']
    image_data = base64.standard_b64encode(image_file.read()).decode('utf-8')
    media_type = image_file.content_type or 'image/jpeg'
 
    api_key = current_app.config.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return jsonify({'error': 'no api key'}), 500
 
    # Claude Vision으로 약물명 추출
    try:
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        payload = {
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 300,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': image_data
                        }
                    },
                    {
                        'type': 'text',
                        'text': '''이 이미지에서 약물/의약품을 식별해주세요.
약물명만 간단하게 답해주세요. 예: 타이레놀, 아스피린, ASPIRIN
만약 약이 아니거나 식별 불가능하면 "알 수 없음" 이라고만 답하세요.
다른 설명은 하지 마세요.'''
                    }
                ]
            }]
        }
        r = req.post('https://api.anthropic.com/v1/messages', headers=headers, json=payload, timeout=15)
        result = r.json()
        drug_name = result['content'][0]['text'].strip()
    except Exception as e:
        return jsonify({'error': 'vision failed: ' + str(e)}), 500
 
    if drug_name == '알 수 없음' or not drug_name:
        return jsonify({'error': 'cannot identify drug', 'raw': drug_name}), 404
 
    return jsonify({'detected_drug': drug_name})
@main.route('/api/drug-shape')
def api_drug_shape():
    import requests as req
    from flask import jsonify, request, current_app

    shape = request.args.get('shape', '')
    color = request.args.get('color', '')
    front = request.args.get('front', '')
    back = request.args.get('back', '')

    api_key = current_app.config.get('MFDS_API_KEY', '')

    try:
        url = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'
        params = {'serviceKey': api_key, 'type': 'json', 'numOfRows': 10}
        if shape: params['drug_shape'] = shape
        if color: params['color_class1'] = color
        if front: params['print_front'] = front
        if back: params['print_back'] = back

        r = req.get(url, params=params, timeout=5)
        data = r.json()
        items = data.get('body', {}).get('items', [])

        result = []
        for item in items:
            result.append({
                'name': item.get('ITEM_NAME', ''),
                'company': item.get('ENTP_NAME', ''),
                'shape': item.get('DRUG_SHAPE', ''),
                'color': item.get('COLOR_CLASS1', ''),
                'etc_otc': item.get('ETC_OTC_CODE', ''),
                'class_name': item.get('CLASS_NAME', ''),
                'chart': item.get('CHART', ''),
                'img_url': item.get('ITEM_IMAGE', ''),
                'print_front': item.get('PRINT_FRONT', ''),
                'print_back': item.get('PRINT_BACK', '')
            })

        return jsonify({'items': result, 'total': data.get('body', {}).get('totalCount', 0)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@main.route('/interaction')
def interaction_page():
    return render_template('interaction.html')


@main.route('/api/interaction')
def api_interaction():
    import pandas as pd
    from flask import jsonify, request

    drug_a = request.args.get('drug_a', '').upper().strip()
    drug_b = request.args.get('drug_b', '').upper().strip()

    if not drug_a or not drug_b:
        return jsonify({'error': 'two drugs required'}), 400

    if drug_a == drug_b:
        return jsonify({'error': 'same drug'}), 400

    df = pd.read_csv('data/processed/processed_faers.csv')

    # 媛??쎈Ъ???섏옄 ID 吏묓빀
    ids_a = set(df[df['drugname'] == drug_a]['primaryid'])
    ids_b = set(df[df['drugname'] == drug_b]['primaryid'])

    # ???쎈Ъ ?숈떆 蹂듭슜 ?섏옄
    ids_both = ids_a & ids_b

    if len(ids_a) == 0:
        return jsonify({'error': f'{drug_a} not found'}), 404
    if len(ids_b) == 0:
        return jsonify({'error': f'{drug_b} not found'}), 404

    if len(ids_both) == 0:
        return jsonify({
            'drug_a': drug_a,
            'drug_b': drug_b,
            'co_occurrence': 0,
            'drug_a_total': len(ids_a),
            'drug_b_total': len(ids_b),
            'risk_score': 0,
            'top_reactions': [],
            'serious_rate': 0,
            'message': 'no co-occurrence found'
        })

    # ?숈떆 蹂듭슜 ?섏옄 ?곗씠??    df_both = df[df['primaryid'].isin(ids_both)]

    # 以묒쬆 寃곌낵 鍮꾩쑉
    serious_outcomes = {'DE', 'HO', 'LT'}
    df_both = df[df['primaryid'].isin(ids_both)]
    serious = df_both[df_both['outc_cod'].isin(serious_outcomes)]['primaryid'].nunique()
    serious_rate = round(serious / len(ids_both) * 100, 1)

    # 二쇱슂 遺?묒슜 top 10
    top_reactions = (
        df_both['pt']
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={'pt': 'reaction', 'count': 'count'})
        .to_dict(orient='records')
    )

    # ?꾪뿕 ?먯닔 (?숈떆蹂듭슜 鍮꾩쑉 湲곕컲)
    co_rate_a = len(ids_both) / len(ids_a)
    co_rate_b = len(ids_both) / len(ids_b)
    risk_score = round((co_rate_a + co_rate_b) / 2 * 100, 1)

    return jsonify({
        'drug_a': drug_a,
        'drug_b': drug_b,
        'co_occurrence': len(ids_both),
        'drug_a_total': len(ids_a),
        'drug_b_total': len(ids_b),
        'serious_rate': serious_rate,
        'risk_score': min(risk_score * 10, 100),
        'top_reactions': top_reactions
    })
