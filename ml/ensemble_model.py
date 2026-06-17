"""
Ensemble 모델: XGBoost + RandomForest 가중 평균(soft voting)
모델 비교 실험에서 RandomForest는 Recall이 높고 XGBoost는 SHAP/배포 적합성이 좋다는
트레이드오프가 확인되어, 두 모델의 예측 확률을 가중 평균해 더 균형 잡힌 성능을 노린다.

사용법: python ml/ensemble_model.py
결과: ml/ensemble_report.json, ml/ensemble_weight_search.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
SMOOTHING_K = 20


def prepare_features(df):
    df = df.dropna(subset=['drugname', 'pt', 'outc_cod', 'primaryid'])
    df['risk'] = df['outc_cod'].apply(lambda x: 1 if x in ['DE', 'HO'] else 0)

    le_drug = LabelEncoder()
    df['drug_encoded'] = le_drug.fit_transform(df['drugname'].str.upper())

    le_reac = LabelEncoder()
    df['reac_encoded'] = le_reac.fit_transform(df['pt'].str.upper())

    df['sex_encoded'] = df['sex'].map({'F': 0, 'M': 1}).fillna(2)
    df['age'] = df['age'].fillna(df['age'].median())

    return df


def add_risk_rate_features(df_train, df_test, k=SMOOTHING_K):
    """Bayesian-smoothed target encoding (train_model_optuna.py와 동일한 방식)."""
    global_rate = df_train['risk'].mean()

    drug_grp = df_train.groupby('drug_encoded')['risk']
    drug_rate = (drug_grp.sum() + k * global_rate) / (drug_grp.count() + k)
    df_train['drug_risk_rate'] = df_train['drug_encoded'].map(drug_rate)
    df_test['drug_risk_rate']  = df_test['drug_encoded'].map(drug_rate).fillna(global_rate)

    reac_grp = df_train.groupby('reac_encoded')['risk']
    reac_rate = (reac_grp.sum() + k * global_rate) / (reac_grp.count() + k)
    df_train['reac_risk_rate'] = df_train['reac_encoded'].map(reac_rate)
    df_test['reac_risk_rate']  = df_test['reac_encoded'].map(reac_rate).fillna(global_rate)

    df_train['drug_reac_key'] = df_train['drug_encoded'].astype(str) + '_' + df_train['reac_encoded'].astype(str)
    df_test['drug_reac_key']  = df_test['drug_encoded'].astype(str) + '_' + df_test['reac_encoded'].astype(str)
    combo_grp = df_train.groupby('drug_reac_key')['risk']
    combo_rate = (combo_grp.sum() + k * global_rate) / (combo_grp.count() + k)
    df_train['combo_risk_rate'] = df_train['drug_reac_key'].map(combo_rate)
    df_test['combo_risk_rate']  = df_test['drug_reac_key'].map(combo_rate).fillna(global_rate)

    return df_train, df_test


def metrics_at(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_risk": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "recall_risk": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "precision_risk": round(precision_score(y_true, y_pred, pos_label=1), 4),
    }


if __name__ == '__main__':
    FEATURES = ['sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    # train/cal/test 3-way split (환자 단위)
    # cal: ensemble 가중치를 정할 held-out 세트 (test와 별개, 가중치 선택 자체의 누수 방지)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, df['risk'], groups=df['primaryid']))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, cal_idx = next(gss2.split(df_train, df_train['risk'], groups=df_train['primaryid']))
    df_fit = df_train.iloc[fit_idx].copy()
    df_cal = df_train.iloc[cal_idx].copy()

    overlap = set(df_fit['primaryid']) & set(df_cal['primaryid']) & set(df_test['primaryid'])
    assert len(overlap) == 0, "Patient overlap detected across fit/cal/test"

    df_fit_feat, df_test_feat = add_risk_rate_features(df_fit.copy(), df_test)
    _, df_cal_feat = add_risk_rate_features(df_fit.copy(), df_cal)

    X_fit, y_fit   = df_fit_feat[FEATURES], df_fit_feat['risk']
    X_cal, y_cal   = df_cal_feat[FEATURES], df_cal_feat['risk']
    X_test, y_test = df_test_feat[FEATURES], df_test_feat['risk']

    print(f"Fit: {len(X_fit):,} / Weight-search(cal): {len(X_cal):,} / Test: {len(X_test):,} (all disjoint by patient)\n")

    scale = (y_fit == 0).sum() / (y_fit == 1).sum()

    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=10, scale_pos_weight=scale,
                         random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
    xgb.fit(X_fit, y_fit)

    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                 random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fit, y_fit)

    # 개별 모델 성능 (test셋)
    xgb_probs_test = xgb.predict_proba(X_test)[:, 1]
    rf_probs_test  = rf.predict_proba(X_test)[:, 1]
    xgb_solo = metrics_at(y_test, xgb_probs_test)
    rf_solo  = metrics_at(y_test, rf_probs_test)
    print(f"\nXGBoost solo (test)      : {xgb_solo}")
    print(f"RandomForest solo (test) : {rf_solo}")

    # 가중치 그리드서치: cal 세트에서 F1 기준 최적 가중치 탐색
    xgb_probs_cal = xgb.predict_proba(X_cal)[:, 1]
    rf_probs_cal  = rf.predict_proba(X_cal)[:, 1]

    weights = np.arange(0.0, 1.01, 0.1)
    search_results = []
    best_w, best_f1 = None, -1
    for w in weights:
        # w = XGBoost 비중, (1-w) = RandomForest 비중
        blended = w * xgb_probs_cal + (1 - w) * rf_probs_cal
        m = metrics_at(y_cal, blended)
        search_results.append({"xgb_weight": round(float(w), 2), **m})
        if m["f1_risk"] > best_f1:
            best_f1 = m["f1_risk"]
            best_w = w

    print(f"\nBest XGBoost weight (by F1 on cal set): {best_w:.2f} (RandomForest weight: {1 - best_w:.2f})")

    # 최적 가중치를 test셋에 적용해 최종 평가
    ensemble_probs_test = best_w * xgb_probs_test + (1 - best_w) * rf_probs_test
    ensemble_result = metrics_at(y_test, ensemble_probs_test)
    print(f"Ensemble (test, weight={best_w:.2f})  : {ensemble_result}")

    # 단순 평균(0.5/0.5)도 같이 비교용으로 기록
    simple_avg_probs_test = 0.5 * xgb_probs_test + 0.5 * rf_probs_test
    simple_avg_result = metrics_at(y_test, simple_avg_probs_test)
    print(f"Ensemble (test, simple 0.5/0.5)        : {simple_avg_result}")

    report = {
        "xgb_solo_test": xgb_solo,
        "rf_solo_test": rf_solo,
        "simple_average_ensemble_test": simple_avg_result,
        "best_weight_xgb": round(float(best_w), 2),
        "best_weight_rf": round(float(1 - best_w), 2),
        "weighted_ensemble_test": ensemble_result,
        "weight_search_on_cal_set": search_results,
        "fit_size": len(X_fit),
        "cal_size": len(X_cal),
        "test_size": len(X_test),
        "note": "Weight search performed on a held-out calibration set, disjoint from both fit and test sets by patient (primaryid)."
    }
    json_path = os.path.join(MODEL_DIR, 'ensemble_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {json_path}")

    # 가중치별 F1 추이 plot
    ws = [r['xgb_weight'] for r in search_results]
    f1s = [r['f1_risk'] for r in search_results]
    plt.figure(figsize=(8, 5))
    plt.plot(ws, f1s, 'o-')
    plt.axvline(x=best_w, color='red', linestyle='--', alpha=0.6, label=f'Best weight={best_w:.2f}')
    plt.xlabel('XGBoost weight (1 - weight = RandomForest weight)')
    plt.ylabel('F1 (risk)')
    plt.title('Ensemble Weight Search (evaluated on calibration set)')
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(MODEL_DIR, 'ensemble_weight_search.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
