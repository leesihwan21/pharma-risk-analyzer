"""
모델 비교 실험: Logistic Regression vs RandomForest vs LightGBM vs XGBoost
같은 train/test split(GroupShuffleSplit, primaryid 기준)으로 공정 비교
사용법: python ml/compare_models.py
결과: ml/model_comparison.json, ml/model_comparison.md

[수정 사항] train_model_optuna.py와 동일하게:
1. risk-rate 피처에 Bayesian 스무딩 적용 (표본 적은 카테고리의 극단값 노이즈 완화)
2. 원시 LabelEncoder ID(drug_encoded, reac_encoded)를 모델 피처에서 제거
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from lightgbm import LGBMClassifier
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


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_risk": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "recall_risk": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "precision_risk": round(precision_score(y_test, y_pred, pos_label=1), 4),
    }


if __name__ == '__main__':
    # drug_encoded / reac_encoded 원시 ID는 더 이상 모델 피처로 쓰지 않음
    FEATURES = ['sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, df['risk'], groups=df['primaryid']))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    df_train, df_test = add_risk_rate_features(df_train, df_test)

    X_train, y_train = df_train[FEATURES], df_train['risk']
    X_test,  y_test  = df_test[FEATURES],  df_test['risk']

    overlap = set(df_train['primaryid']) & set(df_test['primaryid'])
    assert len(overlap) == 0, "Data leakage detected"
    print(f"Train: {len(X_train):,} / Test: {len(X_test):,} (no patient overlap)\n")

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=10, scale_pos_weight=scale,
                                    random_state=RANDOM_STATE, verbosity=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=10, scale_pos_weight=scale,
                                  random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        result = evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(result)
        print(f"  Accuracy={result['accuracy']}  F1={result['f1_risk']}  "
              f"Recall={result['recall_risk']}  Precision={result['precision_risk']}\n")

    results_sorted = sorted(results, key=lambda x: x['f1_risk'], reverse=True)

    json_path = os.path.join(MODEL_DIR, 'model_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results_sorted, f, indent=2)

    md_path = os.path.join(MODEL_DIR, 'model_comparison.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison\n\n")
        f.write(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows ")
        f.write("(GroupShuffleSplit on primaryid, no patient overlap)\n\n")
        f.write("| Model | Accuracy | F1 (risk) | Recall (risk) | Precision (risk) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results_sorted:
            f.write(f"| {r['model']} | {r['accuracy']} | {r['f1_risk']} | "
                    f"{r['recall_risk']} | {r['precision_risk']} |\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print("\nBest model by F1:", results_sorted[0]['model'])
