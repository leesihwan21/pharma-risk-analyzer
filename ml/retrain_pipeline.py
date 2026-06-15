"""
FAERS 분기 데이터 자동 재학습 파이프라인
사용법: python ml/retrain_pipeline.py
       python ml/retrain_pipeline.py --quarter 2025q4  (특정 분기 지정)
스케줄: Windows 작업 스케줄러로 분기마다 자동 실행
"""

import os
import sys
import json
import pickle
import argparse
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

# -- 경로 ----------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
PROC_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(BASE_DIR, 'ml', 'pipeline_log.json')

RANDOM_STATE = 42
FEATURES = ['drug_encoded', 'reac_encoded', 'sex_encoded', 'age',
            'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

# FDA FAERS 다운로드 URL 패턴
FAERS_BASE = "https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip"


# -- 1. 최신 분기 자동 감지 ----------------------------------
def get_latest_quarter():
    now = datetime.now()
    year = now.year
    month = now.month
    if month <= 3:
        quarter = f"{year - 1}q4"
    elif month <= 6:
        quarter = f"{year}q1"
    elif month <= 9:
        quarter = f"{year}q2"
    else:
        quarter = f"{year}q3"
    return quarter


# -- 2. FAERS 데이터 다운로드 --------------------------------
def download_faers(quarter):
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, f"faers_{quarter}.zip")

    if os.path.exists(zip_path):
        print(f"Already downloaded: {zip_path}")
        return zip_path

    url = FAERS_BASE.format(quarter=quarter)
    print(f"Downloading FAERS {quarter} from {url} ...")
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {zip_path} ({os.path.getsize(zip_path) // 1024 // 1024}MB)")
        return zip_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Skipping download — using existing processed_faers.csv")
        return None


# -- 3. 새 분기 데이터 전처리 및 병합 -----------------------
def preprocess_and_merge(zip_path, quarter):
    if zip_path is None:
        print("No new data — retraining on existing data")
        return pd.read_csv(PROC_PATH)

    extract_dir = os.path.join(RAW_DIR, f"faers_{quarter}")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    # FAERS ASCII 파일 찾기
    demo_files = []
    reac_files = []
    outc_files = []
    drug_files = []

    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fl = f.lower()
            if fl.startswith('demo') and fl.endswith('.txt'):
                demo_files.append(os.path.join(root, f))
            elif fl.startswith('reac') and fl.endswith('.txt'):
                reac_files.append(os.path.join(root, f))
            elif fl.startswith('outc') and fl.endswith('.txt'):
                outc_files.append(os.path.join(root, f))
            elif fl.startswith('drug') and fl.endswith('.txt'):
                drug_files.append(os.path.join(root, f))

    if not demo_files:
        print("FAERS ASCII files not found — using existing data")
        return pd.read_csv(PROC_PATH)

    print("Parsing FAERS files...")
    sep = '$'

    demo = pd.concat([pd.read_csv(f, sep=sep, encoding='latin1', on_bad_lines='skip',
                                   usecols=lambda c: c.lower() in ['primaryid', 'age', 'sex'])
                      for f in demo_files], ignore_index=True)
    demo.columns = demo.columns.str.lower()

    reac = pd.concat([pd.read_csv(f, sep=sep, encoding='latin1', on_bad_lines='skip',
                                   usecols=lambda c: c.lower() in ['primaryid', 'pt'])
                      for f in reac_files], ignore_index=True)
    reac.columns = reac.columns.str.lower()

    outc = pd.concat([pd.read_csv(f, sep=sep, encoding='latin1', on_bad_lines='skip',
                                   usecols=lambda c: c.lower() in ['primaryid', 'outc_cod'])
                      for f in outc_files], ignore_index=True)
    outc.columns = outc.columns.str.lower()

    drug = pd.concat([pd.read_csv(f, sep=sep, encoding='latin1', on_bad_lines='skip',
                                   usecols=lambda c: c.lower() in ['primaryid', 'drugname'])
                      for f in drug_files], ignore_index=True)
    drug.columns = drug.columns.str.lower()

    new_df = drug.merge(reac, on='primaryid').merge(outc, on='primaryid').merge(demo, on='primaryid')
    new_df['quarter'] = quarter

    # 기존 데이터와 병합
    existing = pd.read_csv(PROC_PATH)
    if 'quarter' not in existing.columns:
        existing['quarter'] = 'legacy'

    merged = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=['primaryid', 'drugname', 'pt'], keep='last'
    )

    merged.to_csv(PROC_PATH, index=False)
    print(f"Merged: {len(existing):,} + {len(new_df):,} = {len(merged):,} rows")
    return merged


# -- 4. 피처 준비 --------------------------------------------
def prepare_features(df):
    df = df.copy()
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

    return df, le_drug, le_reac


# -- 5. 재학습 -----------------------------------------------
def retrain(df, quarter):
    df, le_drug, le_reac = prepare_features(df)

    X = df[FEATURES]
    y = df['risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Optuna 최적 파라미터 로드 (있으면)
    params_path = os.path.join(MODEL_DIR, 'best_params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            best_params = json.load(f)
        print(f"Loaded best_params.json")
    else:
        best_params = {
            "n_estimators": 320, "max_depth": 3,
            "learning_rate": 0.011, "subsample": 0.94,
            "colsample_bytree": 0.75, "min_child_weight": 10, "gamma": 0.15
        }

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    best_params.update({
        "scale_pos_weight": scale,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
        "tree_method": "hist",
    })

    # MLflow 기록
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pharma-risk-xgboost")

    with mlflow.start_run(run_name=f"retrain_{quarter}"):
        mlflow.log_params({k: v for k, v in best_params.items()
                           if k not in ['scale_pos_weight', 'random_state', 'eval_metric', 'verbosity', 'tree_method']})
        mlflow.log_param("quarter", quarter)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("total_rows", len(df))

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, pos_label=1)
        rec  = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("f1_risk", round(f1, 4))
        mlflow.log_metric("recall_risk", round(rec, 4))
        mlflow.log_metric("precision_risk", round(prec, 4))
        mlflow.xgboost.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    print(f"\nRetrain Results ({quarter})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1(risk) : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(classification_report(y_test, y_pred, target_names=['safe', 'risk']))

    # 모델 저장
    pickle.dump(model,   open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(le_drug, open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'wb'))
    pickle.dump(le_reac, open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'wb'))

    # 파이프라인 로그 저장
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append({
        "quarter": quarter,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(acc, 4),
        "f1_risk": round(f1, 4),
        "recall_risk": round(rec, 4),
        "total_rows": len(df)
    })
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved: model.pkl, pipeline_log.json")
    print(f"MLflow run_id: {run_id}")
    return acc, f1


# -- 메인 ----------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quarter', type=str, default=None,
                        help='FAERS quarter e.g. 2025q4 (default: auto-detect)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, retrain on existing data only')
    args = parser.parse_args()

    quarter = args.quarter or get_latest_quarter()
    print(f"Pipeline start: {quarter}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if args.skip_download:
        df = pd.read_csv(PROC_PATH)
        print(f"Loaded existing data: {len(df):,} rows")
    else:
        zip_path = download_faers(quarter)
        df = preprocess_and_merge(zip_path, quarter)

    acc, f1 = retrain(df, quarter)
    print(f"\nPipeline complete: accuracy={acc:.4f}, f1={f1:.4f}")
