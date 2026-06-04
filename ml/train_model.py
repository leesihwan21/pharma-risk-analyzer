import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

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

    # 🔥 피처 엔지니어링 추가
    # 약물별 위험 비율
    drug_risk_rate = df.groupby('drug_encoded')['risk'].mean()
    df['drug_risk_rate'] = df['drug_encoded'].map(drug_risk_rate)

    # 부작용별 위험 비율
    reac_risk_rate = df.groupby('reac_encoded')['risk'].mean()
    df['reac_risk_rate'] = df['reac_encoded'].map(reac_risk_rate)

    # 약물+부작용 조합 위험 비율
    df['drug_reac_key'] = df['drug_encoded'].astype(str) + '_' + df['reac_encoded'].astype(str)
    combo_risk_rate = df.groupby('drug_reac_key')['risk'].mean()
    df['combo_risk_rate'] = df['drug_reac_key'].map(combo_risk_rate)

    print(f"✅ 피처 준비 완료: {df.shape}")
    print(f"   위험(1): {df['risk'].sum()}건 / 비위험(0): {(df['risk']==0).sum()}건")
    return df, le_drug, le_reac

def train(df, le_drug, le_reac):
    print("\n🤖 XGBoost 모델 학습 중...")

    X = df[['drug_encoded', 'reac_encoded', 'sex_encoded', 'age',
            'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']]
    y = df['risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 정확도: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return model

def save_models(model, le_drug, le_reac):
    print("\n💾 모델 저장 중...")
    pickle.dump(model, open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(le_drug, open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'wb'))
    pickle.dump(le_reac, open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'wb'))
    print("✅ 저장 완료!")

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df, le_drug, le_reac = prepare_features(df)
    model = train(df, le_drug, le_reac)
    save_models(model, le_drug, le_reac)
    print("\n🎉 XGBoost 모델 학습 완료!")