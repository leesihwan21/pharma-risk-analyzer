"""
Drift Monitoring: 쿼터별(2024Q1~2025Q1) 모델 성능 추적
같은 모델(첫 쿼터 데이터로 학습)을 쿼터별 데이터에 적용했을 때 성능 변화를 관찰
데이터 분포가 시간에 따라 변하면(drift) 성능이 얼마나 떨어지는지 확인하는 목적
사용법: python ml/drift_monitoring.py
결과:  ml/drift_report.json, ml/drift_trend.png

[수정 사항]
1. 학습 분기(Q1) 평가 누수 수정: 기존엔 Q1 평가 시 학습에 쓰인 80%까지 포함된
   "전체 Q1"으로 평가해서 Q1 F1이 부풀려져 있었음. 이제는 Q1도 학습에 쓰이지 않은
   held-out 20%로만 평가해서 다른 쿼터와 공정하게 비교됨.
2. risk-rate 피처에 Bayesian 스무딩 적용: 표본 수가 적은 약물/부작용은
   global_rate 쪽으로 끌어당겨서 극단값(0.0/1.0) 노이즈를 줄임.
3. 원시 LabelEncoder ID(drug_encoded, reac_encoded)를 모델 피처에서 제거.
   의미 없는 정수 ID라 트리 모델이 특정 값에 과적합할 위험이 있었음.
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
SMOOTHING_K = 20  # 스무딩 강도: 표본이 이 정도보다 적으면 global_rate 쪽으로 더 끌어당김


def prepare_features(df):
    df = df.dropna(subset=['drugname', 'pt', 'outc_cod', 'primaryid', 'quarter'])
    df['risk'] = df['outc_cod'].apply(lambda x: 1 if x in ['DE', 'HO'] else 0)

    le_drug = LabelEncoder()
    df['drug_encoded'] = le_drug.fit_transform(df['drugname'].str.upper())

    le_reac = LabelEncoder()
    df['reac_encoded'] = le_reac.fit_transform(df['pt'].str.upper())

    df['sex_encoded'] = df['sex'].map({'F': 0, 'M': 1}).fillna(2)
    df['age'] = df['age'].fillna(df['age'].median())

    return df


def add_risk_rate_features(df_train, df_eval, k=SMOOTHING_K):
    """Bayesian-smoothed target encoding.
    표본 수(count)가 적을수록 global_rate 쪽으로 더 많이 끌어당겨서,
    train에 1~2번만 등장한 약물/부작용의 극단값(0.0, 1.0) 노이즈를 줄인다.
    smoothed_rate = (count * mean + k * global_rate) / (count + k)
    """
    global_rate = df_train['risk'].mean()

    drug_grp = df_train.groupby('drug_encoded')['risk']
    drug_rate = (drug_grp.sum() + k * global_rate) / (drug_grp.count() + k)
    df_train['drug_risk_rate'] = df_train['drug_encoded'].map(drug_rate)
    df_eval['drug_risk_rate']  = df_eval['drug_encoded'].map(drug_rate).fillna(global_rate)

    reac_grp = df_train.groupby('reac_encoded')['risk']
    reac_rate = (reac_grp.sum() + k * global_rate) / (reac_grp.count() + k)
    df_train['reac_risk_rate'] = df_train['reac_encoded'].map(reac_rate)
    df_eval['reac_risk_rate']  = df_eval['reac_encoded'].map(reac_rate).fillna(global_rate)

    df_train['drug_reac_key'] = df_train['drug_encoded'].astype(str) + '_' + df_train['reac_encoded'].astype(str)
    df_eval['drug_reac_key']  = df_eval['drug_encoded'].astype(str) + '_' + df_eval['reac_encoded'].astype(str)
    combo_grp = df_train.groupby('drug_reac_key')['risk']
    combo_rate = (combo_grp.sum() + k * global_rate) / (combo_grp.count() + k)
    df_train['combo_risk_rate'] = df_train['drug_reac_key'].map(combo_rate)
    df_eval['combo_risk_rate']  = df_eval['drug_reac_key'].map(combo_rate).fillna(global_rate)

    return df_train, df_eval


if __name__ == '__main__':
    # drug_encoded / reac_encoded 원시 ID는 더 이상 모델 피처로 쓰지 않음
    FEATURES = ['sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    quarters = sorted(df['quarter'].unique())
    print(f"Quarters found: {quarters}\n")

    train_quarter = quarters[0]
    eval_quarters = quarters

    df_train_full = df[df['quarter'] == train_quarter].copy()

    # train 쿼터 내에서 환자 단위로 분리 (fit용 80% / held-out 20%)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, holdout_idx = next(gss.split(df_train_full, df_train_full['risk'], groups=df_train_full['primaryid']))
    df_fit = df_train_full.iloc[fit_idx].copy()
    df_q1_holdout = df_train_full.iloc[holdout_idx].copy()  # 학습에 전혀 쓰이지 않은 진짜 held-out

    overlap = set(df_fit['primaryid']) & set(df_q1_holdout['primaryid'])
    assert len(overlap) == 0, f"Leakage detected in train-quarter holdout split! {len(overlap)} patients overlap"

    print(f"Training model on {train_quarter} only ({len(df_fit):,} rows, held-out {len(df_q1_holdout):,} rows)...")

    results = []
    for q in eval_quarters:
        # 학습 분기는 held-out 20%로만 평가 (다른 분기와 공정하게 비교하기 위함)
        if q == train_quarter:
            df_eval = df_q1_holdout.copy()
        else:
            df_eval = df[df['quarter'] == q].copy()

        df_fit_copy = df_fit.copy()
        df_fit_feat, df_eval_feat = add_risk_rate_features(df_fit_copy, df_eval)

        if q == train_quarter:
            X_fit, y_fit = df_fit_feat[FEATURES], df_fit_feat['risk']
            scale = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = XGBClassifier(n_estimators=200, max_depth=10, scale_pos_weight=scale,
                                   random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
            model.fit(X_fit, y_fit)

        X_eval, y_eval = df_eval_feat[FEATURES], df_eval_feat['risk']
        y_pred = model.predict(X_eval)

        result = {
            "quarter": q,
            "n_rows": len(df_eval),
            "accuracy": round(accuracy_score(y_eval, y_pred), 4),
            "f1": round(f1_score(y_eval, y_pred, pos_label=1), 4),
            "recall": round(recall_score(y_eval, y_pred, pos_label=1), 4),
            "precision": round(precision_score(y_eval, y_pred, pos_label=1), 4),
            "is_train_quarter": (q == train_quarter),
        }
        results.append(result)
        print(f"  {q}: F1={result['f1']}  Recall={result['recall']}  Precision={result['precision']}")

    json_path = os.path.join(MODEL_DIR, 'drift_report.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Drift trend plot
    qs = [r['quarter'] for r in results]
    f1s = [r['f1'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(qs, f1s, 'o-', label='F1 (risk)')
    plt.plot(qs, recalls, 's-', label='Recall (risk)')
    plt.plot(qs, precisions, '^-', label='Precision (risk)')
    plt.axvline(x=train_quarter, color='gray', linestyle='--', alpha=0.5, label=f'Trained on {train_quarter}')
    plt.xlabel('Quarter')
    plt.ylabel('Score')
    plt.title(f'Model Drift: Performance Over Time (trained on {train_quarter}, fair held-out evaluation)')
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(MODEL_DIR, 'drift_trend.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
