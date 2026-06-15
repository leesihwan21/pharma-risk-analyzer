"""
SMOTE 클래스 불균형 처리 + Optuna 최적 파라미터 적용
사용법: python ml/train_model_smote.py
결과:  ml/model.pkl (기존 덮어쓰기)
       ml/smote_report.json (성능 비교 저장)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── 경로 ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(MODEL_DIR, 'best_params.json')

RANDOM_STATE = 42


# ── 피처 준비 ────────────────────────────────────────────────
def prepare_features(df):
    print("🔧 피처 준비 중...")
    df = df.dropna(subset=['drugname', 'pt', 'outc_cod'])
    df['risk'] = df['outc_cod'].apply(lambda x: 1 if x in ['DE', 'HO'] else 0)

    le_drug = LabelEncoder()
    df['drug_encoded'] = le_drug.fit_transform(df['drugname'].str.upper())

    le_reac = LabelEncoder()
    df['reac_encoded'] = le_reac.fit_transform(df['pt'].str.upper())

    df['sex_encoded'] = df['sex'].map({'F': 0, 'M': 1}).fillna(2)
    df['age'] = df['age'].fillna(df['age'].median())

    drug_risk_rate = df.groupby('drug_encoded')['risk'].mean()
    df['drug_risk_rate'] = df['drug_encoded'].map(drug_risk_rate)

    reac_risk_rate = df.groupby('reac_encoded')['risk'].mean()
    df['reac_risk_rate'] = df['reac_encoded'].map(reac_risk_rate)

    df['drug_reac_key'] = df['drug_encoded'].astype(str) + '_' + df['reac_encoded'].astype(str)
    combo_risk_rate = df.groupby('drug_reac_key')['risk'].mean()
    df['combo_risk_rate'] = df['drug_reac_key'].map(combo_risk_rate)

    print(f"✅ 피처 준비 완료: {df.shape}")
    print(f"   위험(1): {df['risk'].sum()}건 / 비위험(0): {(df['risk']==0).sum()}건")
    return df, le_drug, le_reac


# ── 메인 ────────────────────────────────────────────────────
if __name__ == '__main__':
    FEATURES = ['drug_encoded', 'reac_encoded', 'sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    # 1. 데이터 로드
    print("📂 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)
    df, le_drug, le_reac = prepare_features(df)

    X = df[FEATURES]
    y = df['risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 2. SMOTE 적용 (train에만!)
    print("\n⚖️  SMOTE 클래스 불균형 처리 중...")
    print(f"   적용 전 → 위험: {y_train.sum():,} / 비위험: {(y_train==0).sum():,}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"   적용 후 → 위험: {y_train_sm.sum():,} / 비위험: {(y_train_sm==0).sum():,}")

    # 3. Optuna 최적 파라미터 로드
    if os.path.exists(PARAMS_PATH):
        print(f"\n📋 Optuna 최적 파라미터 로드: {PARAMS_PATH}")
        with open(PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
    else:
        print("\n⚠️  best_params.json 없음 → 기본 파라미터 사용")
        best_params = {
            "n_estimators": 320,
            "max_depth": 3,
            "learning_rate": 0.011,
            "subsample": 0.94,
            "colsample_bytree": 0.75,
            "min_child_weight": 10,
            "gamma": 0.15,
        }

    # SMOTE 쓰면 scale_pos_weight 제거 (이미 균형 맞춰졌으니까)
    best_params.pop("scale_pos_weight", None)
    best_params.update({
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
        "tree_method": "hist",
    })

    # 4. 학습
    print("\n🤖 SMOTE 데이터로 모델 학습 중...")
    model = XGBClassifier(**best_params)
    model.fit(X_train_sm, y_train_sm)

    # 5. 평가 (테스트셋은 원본 그대로)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label=1)
    rec = __import__('sklearn.metrics', fromlist=['recall_score']).recall_score(y_test, y_pred, pos_label=1)

    print(f"\n📊 최종 테스트셋 성능 (SMOTE 적용)")
    print(f"   Accuracy      : {acc:.4f}  (기존 0.6933)")
    print(f"   F1(위험)      : {f1:.4f}  (기존 0.6217)")
    print(f"   Recall(위험)  : {rec:.4f}  (기존 0.71)")
    print(classification_report(y_test, y_pred, target_names=['비위험', '위험']))

    # 6. 결과 저장
    report = {
        "before_smote": {"accuracy": 0.6933, "f1_risk": 0.6217, "recall_risk": 0.71},
        "after_smote":  {"accuracy": round(acc, 4), "f1_risk": round(f1, 4), "recall_risk": round(rec, 4)}
    }
    report_path = os.path.join(MODEL_DIR, 'smote_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 성능 비교 저장: {report_path}")

    # 7. 모델 저장
    pickle.dump(model,   open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(le_drug, open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'wb'))
    pickle.dump(le_reac, open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'wb'))
    print("✅ 저장 완료! (model.pkl 덮어쓰기)")

    print("\n🎉 SMOTE 학습 완료!")
