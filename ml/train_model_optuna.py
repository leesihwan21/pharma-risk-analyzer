"""
Optuna + MLflow 실험 추적
사용법: python ml/train_model_optuna.py
결과:  ml/model.pkl, ml/best_params.json
       mlruns/ (MLflow 실험 기록)
UI:    mlflow ui -> http://127.0.0.1:5000

Data Leakage Prevention:
- Patient-level leakage prevented via GroupShuffleSplit on primaryid
- Risk-rate features computed on TRAIN set only, then mapped to test

[수정 사항]
1. risk-rate 피처에 Bayesian 스무딩 적용: 표본 수가 적은 약물/부작용은
   global_rate 쪽으로 끌어당겨서 극단값(0.0/1.0) 노이즈를 줄임.
2. 원시 LabelEncoder ID(drug_encoded, reac_encoded)를 모델 피처에서 제거.
   의미 없는 정수 ID라 트리 모델이 특정 값에 과적합할 위험이 있었고,
   risk-rate 피처가 이미 더 안정적으로 같은 정보를 담고 있음.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

# -- 경로 ----------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# -- 설정 ----------------------------------------------------
N_TRIALS     = 1
CV_FOLDS     = 3
RANDOM_STATE = 42
SMOOTHING_K  = 20  # 스무딩 강도: 표본이 이 정도보다 적으면 global_rate 쪽으로 더 끌어당김
EXPERIMENT_NAME = "pharma-risk-xgboost"


# -- 기본 전처리 (risk_rate 피처는 여기서 만들지 않음) ----------
def prepare_features(df):
    print("Preparing base features...")
    df = df.dropna(subset=['drugname', 'pt', 'outc_cod', 'primaryid'])
    df['risk'] = df['outc_cod'].apply(lambda x: 1 if x in ['DE', 'HO'] else 0)

    le_drug = LabelEncoder()
    df['drug_encoded'] = le_drug.fit_transform(df['drugname'].str.upper())

    le_reac = LabelEncoder()
    df['reac_encoded'] = le_reac.fit_transform(df['pt'].str.upper())

    df['sex_encoded'] = df['sex'].map({'F': 0, 'M': 1}).fillna(2)
    df['age'] = df['age'].fillna(df['age'].median())

    print(f"Done: {df.shape} | Risk(1): {df['risk'].sum()} / Safe(0): {(df['risk']==0).sum()}")
    return df, le_drug, le_reac


# -- Train 기준 risk-rate 피처 생성 (누수 방지 + Bayesian 스무딩) ------------------
def add_risk_rate_features(df_train, df_test, k=SMOOTHING_K):
    """smoothed_rate = (count * mean + k * global_rate) / (count + k)
    표본이 적은 약물/부작용일수록 global_rate 쪽으로 더 강하게 끌어당겨,
    1~2건만 등장한 카테고리가 risk_rate=0.0/1.0 같은 극단값을 갖는 문제를 완화한다.
    """
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


# -- Optuna objective -----------------------------------------
def make_objective(X_train, y_train):
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": scale,
            "random_state":     RANDOM_STATE,
            "eval_metric":      "logloss",
            "verbosity":        0,
            "tree_method":      "hist",
        }

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, pos_label=1))

        return np.mean(scores)

    return objective


# -- 메인 ----------------------------------------------------
if __name__ == '__main__':
    # drug_encoded / reac_encoded 원시 ID는 더 이상 모델 피처로 쓰지 않음
    # (의미 없는 정수 ID라 과적합 위험; risk_rate 피처가 더 안정적으로 같은 정보를 담음)
    FEATURES = ['sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    # 1. 데이터 로드
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df, le_drug, le_reac = prepare_features(df)

    # 2. 환자(primaryid) 단위 split — 데이터 누수 방지 핵심
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, df['risk'], groups=df['primaryid']))
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    # 3. risk-rate 피처는 train으로만 계산 (피처 누수 방지 + 스무딩)
    df_train, df_test = add_risk_rate_features(df_train, df_test)

    X_train, y_train = df_train[FEATURES], df_train['risk']
    X_test,  y_test  = df_test[FEATURES],  df_test['risk']

    print(f"Train: {len(X_train):,} rows ({df_train['primaryid'].nunique():,} unique patients)")
    print(f"Test:  {len(X_test):,} rows ({df_test['primaryid'].nunique():,} unique patients)")

    # 환자 중복 검증 (안전장치)
    overlap = set(df_train['primaryid']) & set(df_test['primaryid'])
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} patients in both sets"
    print("No patient overlap between train/test confirmed (GroupShuffleSplit on primaryid)")

    # 4. MLflow 실험 설정
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 5. Optuna 튜닝
    print(f"\nOptuna tuning ({N_TRIALS} trials, CV={CV_FOLDS}fold)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest F1 (CV): {study.best_value:.4f}")

    # 6. MLflow run 시작
    with mlflow.start_run(run_name="optuna_xgboost_smoothed_te"):

        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("split_method", "GroupShuffleSplit_primaryid")
        mlflow.log_param("target_encoding_smoothing_k", SMOOTHING_K)
        mlflow.log_param("raw_label_id_features", False)

        scale = (y_train == 0).sum() / (y_train == 1).sum()
        final_params = best_params.copy()
        final_params.update({
            "scale_pos_weight": scale,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
            "tree_method": "hist",
        })
        model = XGBClassifier(**final_params)
        model.fit(X_train, y_train)

        # 평가
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, pos_label=1)
        rec  = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("f1_risk", round(f1, 4))
        mlflow.log_metric("recall_risk", round(rec, 4))
        mlflow.log_metric("precision_risk", round(prec, 4))
        mlflow.log_metric("cv_best_f1", round(study.best_value, 4))

        mlflow.xgboost.log_model(model, "model")

        print(f"\nTest Results (patient-level split, no leakage, smoothed target encoding)")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  F1(risk)      : {f1:.4f}")
        print(f"  Recall(risk)  : {rec:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(classification_report(y_test, y_pred, target_names=['safe', 'risk']))

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run_id: {run_id}")

    # 7. 저장
    params_path = os.path.join(MODEL_DIR, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    pickle.dump(model,   open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(le_drug, open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'wb'))
    pickle.dump(le_reac, open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'wb'))
    print("\nSaved: model.pkl, le_drug.pkl, le_reac.pkl, best_params.json")
    print("Done! Run 'mlflow ui' to view results.")
