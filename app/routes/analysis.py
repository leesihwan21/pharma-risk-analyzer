import io
import os
import re
import math
import pickle
import shap
import numpy as np
import pandas as pd
import requests as http_requests

from flask import Blueprint, render_template, jsonify, request, current_app
from app import cache
from app.models import FavoriteDrug

analysis = Blueprint('analysis', __name__)

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

# ── PRR ──────────────────────────────────────
def compute_prr_summary(drugname, df=None):
    """약물의 PRR 신호 탐지 결과 반환 (없으면 None). /api/prr, 즐겨찾기 알림에서 공용 사용."""
    if df is None:
        df = load_df()
    drugname = drugname.upper()

    drug_reports = df[df['drugname'].str.upper() == drugname]
    if len(drug_reports) == 0:
        return None

    other_reports = df[df['drugname'].str.upper() != drugname]
    total_drug = len(drug_reports)
    total_other = len(other_reports)

    if total_other == 0:
        return None

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
        try:
            se = math.sqrt((1/a) - (1/b) + (1/c) - (1/d))
            prr_lower = math.exp(math.log(prr) - 1.96 * se)
            prr_upper = math.exp(math.log(prr) + 1.96 * se)
        except (ValueError, ZeroDivisionError):
            prr_lower = prr_upper = prr

        is_signal = prr >= 2 and a >= 3
        results.append({
            'reaction': reac,
            'drug_count': int(a),
            'drug_total': int(b),
            'other_count': int(c),
            'other_total': int(d),
            'drug_pct': round(a/b*100, 2),
            'other_pct': round(c/d*100, 2),
            'prr': round(prr, 2),
            'prr_lower': round(prr_lower, 2),
            'prr_upper': round(prr_upper, 2),
            'is_signal': is_signal,
            'signal_level': (
                '🔴 강한 신호' if prr >= 5 and a >= 3 else
                '🟡 신호' if prr >= 2 and a >= 3 else
                '⚪ 비신호'
            )
        })

    results.sort(key=lambda x: x['prr'], reverse=True)
    signal_count = sum(1 for r in results if r['is_signal'])
    strong_signal_count = sum(1 for r in results if r['prr'] >= 5 and r['drug_count'] >= 3)

    return {
        'drugname': drugname,
        'total_reports': total_drug,
        'signal_count': signal_count,
        'strong_signal_count': strong_signal_count,
        'results': results
    }

@analysis.route('/prr')
def prr_page():
    return render_template('prr.html')

@analysis.route('/api/prr/<drugname>')
@cache.cached(timeout=600)
def calculate_prr(drugname):
    summary = compute_prr_summary(drugname)
    if summary is None:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname.upper()}'}), 404
    return jsonify(summary)

@analysis.route('/api/favorites/alerts')
def favorites_alerts():
    """즐겨찾기한 약물의 PRR 부작용 신호 알림 (약물감시)"""
    favorites = FavoriteDrug.query.all()
    if not favorites:
        return jsonify({'alerts': []})

    df = load_df()
    alerts = []
    seen = set()
    for fav in favorites:
        drugname = fav.drugname.upper()
        if drugname in seen:
            continue
        seen.add(drugname)

        summary = compute_prr_summary(drugname, df=df)
        if summary is None:
            continue

        top_signals = [r for r in summary['results'] if r['is_signal']][:3]
        if summary['strong_signal_count'] > 0:
            level = 'strong'
        elif summary['signal_count'] > 0:
            level = 'signal'
        else:
            level = 'none'

        alerts.append({
            'drugname': summary['drugname'],
            'signal_count': summary['signal_count'],
            'strong_signal_count': summary['strong_signal_count'],
            'level': level,
            'top_signals': [
                {'reaction': r['reaction'], 'prr': r['prr'], 'signal_level': r['signal_level']}
                for r in top_signals
            ]
        })

    # 강한 신호 → 신호 → 없음 순으로 정렬
    level_order = {'strong': 0, 'signal': 1, 'none': 2}
    alerts.sort(key=lambda a: level_order[a['level']])

    return jsonify({'alerts': alerts})

# ── Trend ─────────────────────────────────────
@analysis.route('/trend')
def trend_page():
    return render_template('trend.html')

@analysis.route('/api/trend')
def api_trend():
    drugname = request.args.get('drug', '').upper()
    if not drugname:
        return jsonify({'error': 'no drug'}), 400
    df = load_df()
    filtered = df[df['drugname'] == drugname]
    if filtered.empty:
        return jsonify({'error': 'not found'}), 404
    trend = filtered.groupby('quarter').size().reset_index(name='count')
    trend = trend.sort_values('quarter')
    return jsonify({'drug': drugname, 'trend': trend.to_dict(orient='records')})

# ── SHAP ──────────────────────────────────────
@analysis.route('/shap')
def shap_page():
    return render_template('shap.html')

def compute_shap(drugname, reaction, age, sex):
    """SHAP 기반 예측+특성기여도 계산. ValueError(message)로 알 수 없는 약물/부작용 처리."""
    drugname = drugname.upper()
    reaction = reaction.upper()

    model, le_drug, le_reac = load_model()
    risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))

    if drugname not in le_drug.classes_:
        raise ValueError('unknown drug: ' + drugname)
    if reaction not in le_reac.classes_:
        raise ValueError('unknown reaction: ' + reaction)

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
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]

    feature_display = {
        'drug': drugname, 'reaction': reaction, 'sex': sex, 'age': age,
        'drug_risk_rate': round(drug_risk_rate, 3),
        'reac_risk_rate': round(reac_risk_rate, 3),
        'combo_risk_rate': round(combo_risk_rate, 3)
    }

    shap_result = [
        {'feature': name, 'value': feature_display[name], 'shap': round(float(sv[i]), 4)}
        for i, name in enumerate(feature_names)
    ]
    shap_result.sort(key=lambda x: abs(x['shap']), reverse=True)

    return {
        'drug': drugname, 'reaction': reaction,
        'prediction': pred,
        'risk_label': 'HIGH RISK' if pred == 1 else 'LOW RISK',
        'probability': {
            'safe': round(float(prob[0]) * 100, 1),
            'risk': round(float(prob[1]) * 100, 1)
        },
        'shap': shap_result
    }

@analysis.route('/api/shap')
def api_shap():
    drugname = request.args.get('drug', '').upper()
    reaction = request.args.get('reaction', '').upper()
    age = float(request.args.get('age', 50))
    sex = request.args.get('sex', 'F')

    try:
        result = compute_shap(drugname, reaction, age, sex)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'model load failed: ' + str(e)}), 500

    return jsonify(result)

FEATURE_LABELS_KO = {
    'drug': '약물 종류',
    'reaction': '부작용 종류',
    'sex': '성별',
    'age': '나이',
    'drug_risk_rate': '해당 약물의 평균 위험률',
    'reac_risk_rate': '해당 부작용의 평균 위험률',
    'combo_risk_rate': '약물-부작용 조합의 위험률',
}

@analysis.route('/api/shap/explain', methods=['POST'])
def api_shap_explain():
    """SHAP 특성 기여도를 근거로 LLM이 예측 결과를 한국어로 설명"""
    data = request.get_json() or {}
    drugname = data.get('drug', '').upper()
    reaction = data.get('reaction', '').upper()
    age = float(data.get('age', 50))
    sex = data.get('sex', 'F')

    try:
        result = compute_shap(drugname, reaction, age, sex)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'model load failed: ' + str(e)}), 500

    top_features = result['shap'][:3]
    lines = []
    for f in top_features:
        label = FEATURE_LABELS_KO.get(f['feature'], f['feature'])
        direction = '위험도를 높이는 방향' if f['shap'] >= 0 else '위험도를 낮추는 방향'
        lines.append(f"- {label} (값: {f['value']}) → {direction}으로 기여 (SHAP {f['shap']:+.3f})")
    shap_lines = '\n'.join(lines)

    prompt = f"""당신은 한국어 임상 데이터 설명 보조원입니다. 반드시 자연스러운 한국어와 약물명 등 필요한 영어 단어만 사용하세요.
절대 규칙:
- 중국어 한자, 힌디어, 일본어, 기타 외국어 문자를 절대 섞지 마세요.
- 아래 SHAP 분석 결과에 없는 의학적 주장을 새로 만들지 마세요.

[예측 결과]
약물: {result['drug']} / 부작용: {result['reaction']}
AI 판정: {result['risk_label']} (안전 {result['probability']['safe']}%, 위험 {result['probability']['risk']}%)

[SHAP 특성 기여도 Top 3 - 영향력 큰 순서]
{shap_lines}

[답변] 위 SHAP 기여도를 근거로, 어떤 요인이 이 예측에 가장 크게 영향을 줬는지 한국어 3문장 이내로 설명하세요. 숫자(SHAP 값)를 그대로 나열하지 말고 의미를 풀어서 설명하세요:"""

    try:
        response = http_requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2', 'prompt': prompt, 'stream': False,
                'options': {'temperature': 0.2, 'top_p': 0.85}
            },
            timeout=60
        )
        explanation = response.json().get('response', '설명 생성 실패')
        explanation = re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]', '', explanation)
    except Exception as e:
        explanation = f'Ollama 오류: {str(e)}'

    return jsonify({
        'drug': result['drug'], 'reaction': result['reaction'],
        'top_features': [
            {'feature': FEATURE_LABELS_KO.get(f['feature'], f['feature']), 'value': f['value'], 'shap': f['shap']}
            for f in top_features
        ],
        'explanation': explanation
    })

# ── Drug Lookup ───────────────────────────────
@analysis.route('/drug-lookup')
def drug_lookup_page():
    return render_template('drug_lookup.html')

@analysis.route('/api/drug-lookup')
def api_drug_lookup():
    drugname = request.args.get('name', '').strip()
    if not drugname:
        return jsonify({'error': 'no name'}), 400

    api_key = current_app.config.get('MFDS_API_KEY', '')
    results = {'korean': None, 'openfda': None}

    try:
        mfds_url = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'
        params = {'serviceKey': api_key, 'item_name': drugname, 'type': 'json', 'numOfRows': 1}
        r = http_requests.get(mfds_url, params=params, timeout=5)
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

    try:
        fda_url = 'https://api.fda.gov/drug/label.json'
        params = {
            'search': f'openfda.brand_name:"{drugname.upper()}" OR openfda.generic_name:"{drugname.upper()}"',
            'limit': 1
        }
        r = http_requests.get(fda_url, params=params, timeout=5)
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

@analysis.route('/api/drug-shape')
def api_drug_shape():
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

        r = http_requests.get(url, params=params, timeout=5)
        data = r.json()
        items = data.get('body', {}).get('items', [])

        result = [{
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
        } for item in items]

        return jsonify({'items': result, 'total': data.get('body', {}).get('totalCount', 0)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Interaction ───────────────────────────────
@analysis.route('/interaction')
def interaction_page():
    return render_template('interaction.html')

@analysis.route('/api/interaction')
def api_interaction():
    drug_a = request.args.get('drug_a', '').upper().strip()
    drug_b = request.args.get('drug_b', '').upper().strip()

    if not drug_a or not drug_b:
        return jsonify({'error': 'two drugs required'}), 400
    if drug_a == drug_b:
        return jsonify({'error': 'same drug'}), 400

    df = pd.read_csv(DATA_PATH)
    ids_a = set(df[df['drugname'] == drug_a]['primaryid'])
    ids_b = set(df[df['drugname'] == drug_b]['primaryid'])
    ids_both = ids_a & ids_b

    if len(ids_a) == 0:
        return jsonify({'error': f'{drug_a} not found'}), 404
    if len(ids_b) == 0:
        return jsonify({'error': f'{drug_b} not found'}), 404

    if len(ids_both) == 0:
        return jsonify({
            'drug_a': drug_a, 'drug_b': drug_b,
            'co_occurrence': 0, 'drug_a_total': len(ids_a),
            'drug_b_total': len(ids_b), 'risk_score': 0,
            'top_reactions': [], 'serious_rate': 0,
            'message': 'no co-occurrence found'
        })

    df_both = df[df['primaryid'].isin(ids_both)]
    serious_outcomes = {'DE', 'HO', 'LT'}
    serious = df_both[df_both['outc_cod'].isin(serious_outcomes)]['primaryid'].nunique()
    serious_rate = round(serious / len(ids_both) * 100, 1)

    top_reactions = (
        df_both['pt'].value_counts().head(10).reset_index()
        .rename(columns={'pt': 'reaction', 'count': 'count'})
        .to_dict(orient='records')
    )

    co_rate_a = len(ids_both) / len(ids_a)
    co_rate_b = len(ids_both) / len(ids_b)
    risk_score = round((co_rate_a + co_rate_b) / 2 * 100, 1)

    return jsonify({
        'drug_a': drug_a, 'drug_b': drug_b,
        'co_occurrence': len(ids_both),
        'drug_a_total': len(ids_a), 'drug_b_total': len(ids_b),
        'serious_rate': serious_rate,
        'risk_score': min(risk_score * 10, 100),
        'top_reactions': top_reactions
    })

# ── Polypharmacy (다중 약물 상호작용) ──────────
@analysis.route('/polypharmacy')
def polypharmacy_page():
    return render_template('polypharmacy.html')

@analysis.route('/api/polypharmacy')
def api_polypharmacy():
    drugs_param = request.args.get('drugs', '')
    drugs = []
    for d in drugs_param.split(','):
        d = d.strip().upper()
        if d and d not in drugs:
            drugs.append(d)

    if len(drugs) < 2:
        return jsonify({'error': '약물 2개 이상을 입력하세요 (쉼표로 구분)'}), 400
    if len(drugs) > 5:
        return jsonify({'error': '최대 5개 약물까지 분석할 수 있습니다'}), 400

    df = pd.read_csv(DATA_PATH)
    serious_outcomes = {'DE', 'HO', 'LT'}

    id_sets = {}
    totals = {}
    for d in drugs:
        ids = set(df[df['drugname'] == d]['primaryid'])
        if len(ids) == 0:
            return jsonify({'error': f'{d}: 데이터에서 찾을 수 없습니다'}), 404
        id_sets[d] = ids
        totals[d] = len(ids)

    # 약물쌍(pairwise) 위험도
    pairs = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = drugs[i], drugs[j]
            both = id_sets[a] & id_sets[b]
            if both:
                df_both = df[df['primaryid'].isin(both)]
                serious = df_both[df_both['outc_cod'].isin(serious_outcomes)]['primaryid'].nunique()
                serious_rate = round(serious / len(both) * 100, 1)
                co_rate_a = len(both) / totals[a]
                co_rate_b = len(both) / totals[b]
                risk_score = min(round((co_rate_a + co_rate_b) / 2 * 100 * 10, 1), 100)
            else:
                serious_rate, risk_score = 0, 0
            pairs.append({
                'drug_a': a, 'drug_b': b,
                'co_occurrence': len(both),
                'serious_rate': serious_rate,
                'risk_score': risk_score
            })

    # 전체 약물 동시 복용 (교집합)
    all_ids = set.intersection(*id_sets.values())
    if all_ids:
        df_all = df[df['primaryid'].isin(all_ids)]
        serious_all = df_all[df_all['outc_cod'].isin(serious_outcomes)]['primaryid'].nunique()
        overall_serious_rate = round(serious_all / len(all_ids) * 100, 1)
        top_reactions = (
            df_all['pt'].value_counts().head(10).reset_index()
            .rename(columns={'pt': 'reaction', 'count': 'count'})
            .to_dict(orient='records')
        )
    else:
        overall_serious_rate = 0
        top_reactions = []

    overall_risk = round(sum(p['risk_score'] for p in pairs) / len(pairs), 1) if pairs else 0
    high_risk_pairs = sorted(
        [p for p in pairs if p['co_occurrence'] > 0],
        key=lambda p: p['risk_score'], reverse=True
    )

    return jsonify({
        'drugs': drugs,
        'totals': totals,
        'pairs': pairs,
        'high_risk_pairs': high_risk_pairs,
        'overall': {
            'co_occurrence': len(all_ids),
            'serious_rate': overall_serious_rate,
            'top_reactions': top_reactions,
            'risk_score': overall_risk
        }
    })

# ── Dosage ────────────────────────────────────
@analysis.route('/dosage')
def dosage_page():
    return render_template('dosage.html')

@analysis.route('/api/dosage/crcl', methods=['POST'])
def api_crcl():
    data = request.get_json()
    age = float(data.get('age', 0))
    weight = float(data.get('weight', 0))
    creatinine = float(data.get('creatinine', 0))
    sex = data.get('sex', 'M')

    if not all([age, weight, creatinine]):
        return jsonify({'error': 'missing values'}), 400

    crcl = ((140 - age) * weight) / (72 * creatinine)
    if sex == 'F':
        crcl *= 0.85

    if crcl >= 90:
        stage, dose_adj, color = 'Normal (G1)', 'No dose adjustment required', 'green'
    elif crcl >= 60:
        stage, dose_adj, color = 'Mild reduction (G2)', 'Dose adjustment may be required', 'yellow'
    elif crcl >= 30:
        stage, dose_adj, color = 'Moderate reduction (G3)', 'Reduce dose by 50-75%', 'orange'
    elif crcl >= 15:
        stage, dose_adj, color = 'Severe reduction (G4)', 'Reduce dose by 25-50%', 'red'
    else:
        stage, dose_adj, color = 'Renal failure (G5)', 'Nephrotoxic drugs contraindicated', 'darkred'

    return jsonify({'crcl': round(crcl, 1), 'stage': stage, 'dose_adj': dose_adj, 'color': color})

@analysis.route('/api/dosage/pediatric', methods=['POST'])
def api_pediatric():
    data = request.get_json()
    adult_dose = float(data.get('adult_dose', 0))
    age = float(data.get('age', 0))
    weight = float(data.get('weight', 0))
    height = float(data.get('height', 0))

    if not adult_dose:
        return jsonify({'error': 'missing values'}), 400

    import math
    results = {}
    if weight:
        results['clark'] = round(adult_dose * weight / 70, 2)
    if age:
        results['young'] = round(adult_dose * age / (age + 12), 2)
    if weight and height:
        bsa = math.sqrt((height * weight) / 3600)
        results['bsa'] = round(adult_dose * bsa / 1.73, 2)
        results['bsa_value'] = round(bsa, 2)

    return jsonify(results)

@analysis.route('/api/dosage/bsa', methods=['POST'])
def api_bsa():
    import math
    data = request.get_json()
    weight = float(data.get('weight', 0))
    height = float(data.get('height', 0))
    dose_per_m2 = float(data.get('dose_per_m2', 0))

    if not all([weight, height]):
        return jsonify({'error': 'missing values'}), 400

    bsa = math.sqrt((height * weight) / 3600)
    total_dose = round(bsa * dose_per_m2, 2) if dose_per_m2 else None
    bsa_dubois = 0.007184 * (height ** 0.725) * (weight ** 0.425)

    return jsonify({
        'bsa_mosteller': round(bsa, 3),
        'bsa_dubois': round(bsa_dubois, 3),
        'total_dose': total_dose
    })

# ── Drug Vision ───────────────────────────────
@analysis.route('/api/drug-vision', methods=['POST'])
def api_drug_vision():
    import base64
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400

    image_file = request.files['image']
    image_data = base64.standard_b64encode(image_file.read()).decode('utf-8')
    media_type = image_file.content_type or 'image/jpeg'
    api_key = current_app.config.get('ANTHROPIC_API_KEY', '')

    if not api_key:
        return jsonify({'error': 'no api key'}), 500

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
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': image_data}},
                    {'type': 'text', 'text': '이 이미지에서 약물/의약품을 식별해주세요.\n약물명만 간단하게 답해주세요.\n만약 약이 아니거나 식별 불가능하면 "알 수 없음" 이라고만 답하세요.'}
                ]
            }]
        }
        r = http_requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=payload, timeout=15)
        result = r.json()
        drug_name = result['content'][0]['text'].strip()
    except Exception as e:
        return jsonify({'error': 'vision failed: ' + str(e)}), 500

    if drug_name == '알 수 없음' or not drug_name:
        return jsonify({'error': 'cannot identify drug', 'raw': drug_name}), 404

    return jsonify({'detected_drug': drug_name})

@analysis.route('/api/ebgm/<drugname>')
@cache.cached(timeout=600)
def calculate_ebgm(drugname):
    df = load_df()
    drugname = drugname.upper()
    drug_reports = df[df['drugname'].str.upper() == drugname]
    if len(drug_reports) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname}'}), 404

    total_reports = len(df)
    total_drug = len(drug_reports)
    top_reactions = drug_reports['pt'].value_counts().head(20).index.tolist()

    results = []
    for reac in top_reactions:
        # 2x2 분할표
        a = len(drug_reports[drug_reports['pt'] == reac])          # 약물+반응
        b = total_drug - a                                           # 약물+반응없음
        c = len(df[(df['drugname'].str.upper() != drugname) & (df['pt'] == reac)])  # 타약물+반응
        d = total_reports - a - b - c                               # 타약물+반응없음

        if a == 0 or (a + b) == 0 or (a + c) == 0:
            continue

        # 기대값 (Expected)
        E = (a + b) * (a + c) / total_reports
        if E == 0:
            continue

        # ROR (Reporting Odds Ratio) - EBGM 근사
        if b == 0 or c == 0:
            continue
        ror = (a * d) / (b * c) if (b * c) > 0 else 0

        # EBGM 계산 (베이지안 보정)
        # 사전분포 파라미터 (FDA MGPS 기준)
        alpha1, beta1 = 0.5, 0.5   # 신호 성분
        alpha2, beta2 = 2.0, 10.0  # 배경 성분
        w = 0.1  # 혼합 가중치

        # 사후 기대값 계산
        ebgm = (alpha1 + a) / (beta1 + E) * w + (alpha2 + a) / (beta2 + E) * (1 - w)

        # EBGM 95% 신뢰구간 (근사)
        try:
            se_log = math.sqrt(1/a + 1/E)
            ebgm_lower = math.exp(math.log(max(ebgm, 0.001)) - 1.96 * se_log)
            ebgm_upper = math.exp(math.log(max(ebgm, 0.001)) + 1.96 * se_log)
        except (ValueError, ZeroDivisionError):
            ebgm_lower = ebgm_upper = ebgm

        # 신호 기준: EB05 (하한 95%) >= 2
        eb05 = ebgm_lower
        is_signal = eb05 >= 2 and a >= 3

        results.append({
            'reaction': reac,
            'drug_count': int(a),
            'expected': round(E, 2),
            'ror': round(ror, 2),
            'ebgm': round(ebgm, 3),
            'eb05': round(ebgm_lower, 3),
            'eb95': round(ebgm_upper, 3),
            'is_signal': is_signal,
            'signal_level': (
                '강한 신호' if eb05 >= 5 and a >= 3 else
                '신호' if eb05 >= 2 and a >= 3 else
                '비신호'
            )
        })

    results.sort(key=lambda x: x['ebgm'], reverse=True)
    signal_count = sum(1 for r in results if r['is_signal'])

    return jsonify({
        'drugname': drugname,
        'total_reports': int(total_drug),
        'signal_count': signal_count,
        'results': results,
        'method': 'EBGM (Empirical Bayes Geometric Mean) - FDA MGPS 근사'
    })

# MedDRA SOC 매핑 딕셔너리 (상위 빈출 PT 기준)
MEDDRA_SOC_MAP = {
    # General disorders
    'FATIGUE': 'General disorders', 'MALAISE': 'General disorders',
    'PYREXIA': 'General disorders', 'PAIN': 'General disorders',
    'SWELLING': 'General disorders', 'OEDEMA': 'General disorders',
    'GENERAL PHYSICAL HEALTH DETERIORATION': 'General disorders',
    'ASTHENIA': 'General disorders', 'CHILLS': 'General disorders',

    # Gastrointestinal disorders
    'NAUSEA': 'Gastrointestinal disorders', 'VOMITING': 'Gastrointestinal disorders',
    'DIARRHOEA': 'Gastrointestinal disorders', 'ABDOMINAL DISCOMFORT': 'Gastrointestinal disorders',
    'ABDOMINAL PAIN': 'Gastrointestinal disorders', 'CONSTIPATION': 'Gastrointestinal disorders',
    'DYSPEPSIA': 'Gastrointestinal disorders', 'STOMATITIS': 'Gastrointestinal disorders',

    # Musculoskeletal disorders
    'ARTHRALGIA': 'Musculoskeletal disorders', 'ARTHROPATHY': 'Musculoskeletal disorders',
    'JOINT SWELLING': 'Musculoskeletal disorders', 'MOBILITY DECREASED': 'Musculoskeletal disorders',
    'MYALGIA': 'Musculoskeletal disorders', 'BACK PAIN': 'Musculoskeletal disorders',
    'BONE PAIN': 'Musculoskeletal disorders',

    # Nervous system disorders
    'HEADACHE': 'Nervous system disorders', 'DIZZINESS': 'Nervous system disorders',
    'PARAESTHESIA': 'Nervous system disorders', 'NEUROPATHY PERIPHERAL': 'Nervous system disorders',
    'TREMOR': 'Nervous system disorders', 'SOMNOLENCE': 'Nervous system disorders',

    # Respiratory disorders
    'DYSPNOEA': 'Respiratory disorders', 'COUGH': 'Respiratory disorders',
    'PNEUMONIA': 'Respiratory disorders', 'PULMONARY EMBOLISM': 'Respiratory disorders',
    'RHINORRHOEA': 'Respiratory disorders',

    # Skin disorders
    'RASH': 'Skin disorders', 'PRURITUS': 'Skin disorders',
    'ALOPECIA': 'Skin disorders', 'URTICARIA': 'Skin disorders',
    'ERYTHEMA': 'Skin disorders', 'DERMATITIS': 'Skin disorders',

    # Investigations (lab)
    'HEPATIC ENZYME INCREASED': 'Investigations',
    'BLOOD CHOLESTEROL INCREASED': 'Investigations',
    'WEIGHT INCREASED': 'Investigations', 'WEIGHT DECREASED': 'Investigations',
    'ALANINE AMINOTRANSFERASE INCREASED': 'Investigations',
    'BLOOD CREATININE INCREASED': 'Investigations',

    # Immune system disorders
    'DRUG HYPERSENSITIVITY': 'Immune system disorders',
    'HYPERSENSITIVITY': 'Immune system disorders',
    'ANAPHYLACTIC REACTION': 'Immune system disorders',

    # Vascular disorders
    'HYPERTENSION': 'Vascular disorders', 'HYPOTENSION': 'Vascular disorders',
    'FLUSHING': 'Vascular disorders', 'DEEP VEIN THROMBOSIS': 'Vascular disorders',

    # Infections
    'URINARY TRACT INFECTION': 'Infections and infestations',
    'NASOPHARYNGITIS': 'Infections and infestations',
    'UPPER RESPIRATORY TRACT INFECTION': 'Infections and infestations',
    'PNEUMONIA': 'Infections and infestations',

    # Musculoskeletal (disease)
    'RHEUMATOID ARTHRITIS': 'Musculoskeletal disorders',
    'SYSTEMIC LUPUS ERYTHEMATOSUS': 'Immune system disorders',

    # Social/admin
    'OFF LABEL USE': 'Social circumstances',
    'DRUG INEFFECTIVE': 'General disorders',
    'DRUG INTOLERANCE': 'General disorders',
    'CONDITION AGGRAVATED': 'General disorders',
    'PRODUCT USE IN UNAPPROVED INDICATION': 'Social circumstances',
    'INFUSION RELATED REACTION': 'General disorders',
}

@analysis.route('/api/soc/<drugname>')
@cache.cached(timeout=600)
def soc_analysis(drugname):
    df = load_df()
    drugname = drugname.upper()
    result = df[df['drugname'].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({'error': f'약물을 찾을 수 없어요: {drugname}'}), 404

    top_reac = result['pt'].value_counts().head(50)
    soc_counts = {}
    mapped_reactions = []

    for reac, cnt in top_reac.items():
        soc = MEDDRA_SOC_MAP.get(reac.upper(), 'Other')
        soc_counts[soc] = soc_counts.get(soc, 0) + cnt
        mapped_reactions.append({
            'pt': reac,
            'count': int(cnt),
            'pct': round(cnt / len(result) * 100, 2),
            'soc': soc
        })

    soc_summary = sorted([
        {'soc': k, 'count': v, 'pct': round(v / len(result) * 100, 1)}
        for k, v in soc_counts.items()
    ], key=lambda x: x['count'], reverse=True)

    return jsonify({
        'drugname': drugname,
        'total_reports': len(result),
        'soc_summary': soc_summary,
        'reactions': mapped_reactions,
        'mapped_count': sum(1 for r in mapped_reactions if r['soc'] != 'Other'),
        'note': 'MedDRA SOC 매핑 (빈출 PT 기준 수동 매핑, 포트폴리오용)'
    })