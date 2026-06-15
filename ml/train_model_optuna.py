"""
Optuna + MLflow ?ㅽ뿕 異붿쟻
?ъ슜踰? python ml/train_model_optuna.py
寃곌낵:  ml/model.pkl, ml/best_params.json
       mlruns/ (MLflow ?ㅽ뿕 湲곕줉)
UI:    mlflow ui -> http://127.0.0.1:5000
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

# -- 寃쎈줈 ----------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_faers.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# -- ?ㅼ젙 ----------------------------------------------------
N_TRIALS     = 1
CV_FOLDS     = 3
RANDOM_STATE = 42
EXPERIMENT_NAME = "pharma-risk-xgboost"


# -- ?쇱쿂 以鍮?------------------------------------------------
def prepare_features(df):
    print("Preparing features...")
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

    print(f"Done: {df.shape} | Risk(1): {df['risk'].sum()} / Safe(0): {(df['risk']==0).sum()}")
    return df, le_drug, le_reac


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


# -- 硫붿씤 ----------------------------------------------------
if __name__ == '__main__':
    FEATURES = ['drug_encoded', 'reac_encoded', 'sex_encoded', 'age',
                'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate']

    # 1. ?곗씠??濡쒕뱶
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df, le_drug, le_reac = prepare_features(df)

    X = df[FEATURES]
    y = df['risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 2. MLflow ?ㅽ뿕 ?ㅼ젙
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. Optuna ?쒕떇
    print(f"\nOptuna tuning ({N_TRIALS} trials, CV={CV_FOLDS}fold)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest F1 (CV): {study.best_value:.4f}")

    # 4. MLflow run ?쒖옉
    with mlflow.start_run(run_name="optuna_xgboost"):

        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # 理쒖쟻 ?뚮씪誘명꽣濡??ы븰??        scale = (y_train == 0).sum() / (y_train == 1).sum()
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

        # ?됯?
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

        print(f"\nTest Results")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  F1(risk)      : {f1:.4f}")
        print(f"  Recall(risk)  : {rec:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(classification_report(y_test, y_pred, target_names=['safe', 'risk']))

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run_id: {run_id}")

    # 5. ???    params_path = os.path.join(MODEL_DIR, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    pickle.dump(model,   open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
    pickle.dump(le_drug, open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'wb'))
    pickle.dump(le_reac, open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'wb'))
    print("\nSaved: model.pkl, le_drug.pkl, le_reac.pkl, best_params.json")
    print("Done! Run 'mlflow ui' to view results.")
