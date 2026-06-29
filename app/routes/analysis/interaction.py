"""
app/routes/analysis/interaction.py
약물 상호작용(2약물), Polypharmacy(다중 약물)
"""
from flask import jsonify, render_template, request

from . import analysis
from ._common import load_df

SERIOUS_OUTCOMES = {'DE', 'HO', 'LT'}


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

    df     = load_df()
    ids_a  = set(df[df['drugname'] == drug_a]['primaryid'])
    ids_b  = set(df[df['drugname'] == drug_b]['primaryid'])
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

    df_both      = df[df['primaryid'].isin(ids_both)]
    serious      = df_both[df_both['outc_cod'].isin(SERIOUS_OUTCOMES)]['primaryid'].nunique()
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

    df = load_df()
    id_sets = {}
    totals  = {}
    for d in drugs:
        ids = set(df[df['drugname'] == d]['primaryid'])
        if len(ids) == 0:
            return jsonify({'error': f'{d}: 데이터에서 찾을 수 없습니다'}), 404
        id_sets[d] = ids
        totals[d]  = len(ids)

    pairs = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = drugs[i], drugs[j]
            both = id_sets[a] & id_sets[b]
            if both:
                df_both      = df[df['primaryid'].isin(both)]
                serious      = df_both[df_both['outc_cod'].isin(SERIOUS_OUTCOMES)]['primaryid'].nunique()
                serious_rate = round(serious / len(both) * 100, 1)
                co_rate_a    = len(both) / totals[a]
                co_rate_b    = len(both) / totals[b]
                risk_score   = min(round((co_rate_a + co_rate_b) / 2 * 100 * 10, 1), 100)
            else:
                serious_rate, risk_score = 0, 0
            pairs.append({
                'drug_a': a, 'drug_b': b,
                'co_occurrence': len(both),
                'serious_rate': serious_rate,
                'risk_score': risk_score
            })

    all_ids = set.intersection(*id_sets.values())
    if all_ids:
        df_all               = df[df['primaryid'].isin(all_ids)]
        serious_all          = df_all[df_all['outc_cod'].isin(SERIOUS_OUTCOMES)]['primaryid'].nunique()
        overall_serious_rate = round(serious_all / len(all_ids) * 100, 1)
        top_reactions        = (
            df_all['pt'].value_counts().head(10).reset_index()
            .rename(columns={'pt': 'reaction', 'count': 'count'})
            .to_dict(orient='records')
        )
    else:
        overall_serious_rate = 0
        top_reactions        = []

    overall_risk    = round(sum(p['risk_score'] for p in pairs) / len(pairs), 1) if pairs else 0
    high_risk_pairs = sorted(
        [p for p in pairs if p['co_occurrence'] > 0],
        key=lambda p: p['risk_score'], reverse=True
    )

    return jsonify({
        'drugs': drugs, 'totals': totals, 'pairs': pairs,
        'high_risk_pairs': high_risk_pairs,
        'overall': {
            'co_occurrence': len(all_ids),
            'serious_rate': overall_serious_rate,
            'top_reactions': top_reactions,
            'risk_score': overall_risk
        }
    })
