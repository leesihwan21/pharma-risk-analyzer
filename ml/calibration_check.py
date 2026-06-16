"""
Calibration 검증: XGBoost 모델의 확률 예측이 실제 확률과 얼마나 일치하는지 확인
의료 AI에서 "고위험 87%"라는 출력이 실제로 87% 확률을 의미하는지 검증하는 과정
사용법: python ml/calibration_check.py
결과: ml/calibration_curve.png, ml/calibration_report.json
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


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


def add_risk_rate_features(df_train, df_test):
    global_rate = df_train['risk'].mean()

    drug_rate = df_train.groupby('drug_encoded')['risk'].mean()
    df_train['drug_risk_rate'] = df_train['drug_encoded'].map(drug_rate)
    df_test['drug_risk_rate']  = df_test['drug_encoded'].map(drug_rate).fillna(global_rate)

    reac_rate = df_train.groupby('reac_encoded')['risk'].mean()
    df_train['reac_risk_rate'] = df_train['reac_encoded'].map(reac_rate)
    df_test['reac_risk_rate']  = df_test['reac_encoded'].map(reac_rate).fillna(global_rate)

    df_train['drug_reac_key'] = df_train['drug_encoded'].astype(str) + '_' + df_train['reac_encoded'].astype(str)
    df_test['drug_reac_key']  = df_test['drug_encoded'].astype(str) + '_' + df_test['reac_encoded'].astype(str)
    combo_rate = df_train.groupby('drug_reac_key')['risk'].mean()
    df_train['combo_risk_rate'] = df_train['drug_reac_key'].map(combo_rate)
    df_test['combo_risk_rate']  = df_test['drug_reac_key'].map(combo_rate).fillna(global_rate)

    return df_train, df_test


if __name__ == '__main__':
    FEATURES = ['drug_encoded', 'reac_encoded', 'sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    # 1차 split: train(전체) vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, df['risk'], groups=df['primaryid']))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    df_train, df_test = add_risk_rate_features(df_train, df_test)

    X_test, y_test = df_test[FEATURES], df_test['risk']

    # 2차 split: train -> fit(모델 학습용) + cal(calibration 전용, test와 별개)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, cal_idx = next(gss2.split(df_train, df_train['risk'], groups=df_train['primaryid']))
    df_fit = df_train.iloc[fit_idx]
    df_cal = df_train.iloc[cal_idx]

    X_fit, y_fit = df_fit[FEATURES], df_fit['risk']
    X_cal, y_cal = df_cal[FEATURES], df_cal['risk']

    overlap_check = set(df_fit['primaryid']) & set(df_cal['primaryid']) & set(df_test['primaryid'])
    assert len(overlap_check) == 0, "Patient overlap detected across fit/cal/test"
    print(f"Fit: {len(X_fit):,} / Calibration: {len(X_cal):,} / Test: {len(X_test):,} (all disjoint by patient)\n")

    scale = (y_fit == 0).sum() / (y_fit == 1).sum()

    print("Training base XGBoost...")
    base_model = XGBClassifier(n_estimators=200, max_depth=10, scale_pos_weight=scale,
                                random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
    base_model.fit(X_fit, y_fit)
    base_probs = base_model.predict_proba(X_test)[:, 1]

    print("Calibrating with isotonic regression (fit on held-out calibration set)...")
    frozen = FrozenEstimator(base_model)
    calibrated_model = CalibratedClassifierCV(frozen, method='isotonic')
    calibrated_model.fit(X_cal, y_cal)
    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]

    brier_base = brier_score_loss(y_test, base_probs)
    brier_calibrated = brier_score_loss(y_test, calibrated_probs)

    print(f"\nBrier Score (base XGBoost)       : {brier_base:.4f}")
    print(f"Brier Score (calibrated isotonic): {brier_calibrated:.4f}")
    print("(Lower is better; 0 = perfect calibration)")

    frac_base, mean_base = calibration_curve(y_test, base_probs, n_bins=10)
    frac_cal,  mean_cal  = calibration_curve(y_test, calibrated_probs, n_bins=10)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(mean_base, frac_base, 's-', label=f'XGBoost (Brier={brier_base:.4f})')
    plt.plot(mean_cal, frac_cal, 'o-', label=f'Calibrated (Brier={brier_calibrated:.4f})')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve: Predicted Risk vs Actual Risk')
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(MODEL_DIR, 'calibration_curve.png')
    plt.savefig(png_path, dpi=150)
    print(f"\nSaved: {png_path}")

    report = {
        "brier_score_base": round(float(brier_base), 4),
        "brier_score_calibrated": round(float(brier_calibrated), 4),
        "fit_size": len(X_fit),
        "calibration_size": len(X_cal),
        "test_size": len(X_test),
        "note": "Calibration fit on a held-out patient-disjoint calibration set, separate from both training and test data."
    }
    json_path = os.path.join(MODEL_DIR, 'calibration_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {json_path}")