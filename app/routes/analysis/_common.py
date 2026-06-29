"""
app/routes/analysis/_common.py
공통 유틸리티: load_df, load_model, load_explainer
"""

import os
import pickle
import pandas as pd

from flask import current_app


def load_df() -> pd.DataFrame:
    return pd.read_csv(current_app.config["DATA_PATH"])


def load_model():
    model_dir = current_app.config["MODEL_DIR"]
    model = pickle.load(open(os.path.join(model_dir, "model.pkl"), "rb"))
    le_drug = pickle.load(open(os.path.join(model_dir, "le_drug.pkl"), "rb"))
    le_reac = pickle.load(open(os.path.join(model_dir, "le_reac.pkl"), "rb"))
    return model, le_drug, le_reac


def load_explainer():
    model_dir = current_app.config["MODEL_DIR"]
    return pickle.load(open(os.path.join(model_dir, "explainer.pkl"), "rb"))
