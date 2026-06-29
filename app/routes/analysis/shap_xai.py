"""
app/routes/analysis/shap_xai.py
SHAP 기반 XAI 설명, LIME
"""

import os
import re
import pickle

import numpy as np
import requests as http_requests

from flask import jsonify, render_template, request, current_app

from . import analysis
from ._common import load_model, load_explainer

FEATURE_LABELS_KO = {
    "drug": "약물 종류",
    "reaction": "부작용 종류",
    "sex": "성별",
    "age": "나이",
    "drug_risk_rate": "해당 약물의 평균 위험률",
    "reac_risk_rate": "해당 부작용의 평균 위험률",
    "combo_risk_rate": "약물-부작용 조합의 위험률",
}


# ── 공용 함수 ─────────────────────────────────


def compute_shap(drugname: str, reaction: str, age: float, sex: str) -> dict:
    """SHAP 기반 예측+특성기여도 계산."""
    drugname = drugname.upper()
    reaction = reaction.upper()

    model, le_drug, le_reac = load_model()
    risk_rates = pickle.load(
        open(os.path.join(current_app.config["MODEL_DIR"], "risk_rates.pkl"), "rb")
    )

    if drugname not in le_drug.classes_:
        raise ValueError("unknown drug: " + drugname)
    if reaction not in le_reac.classes_:
        raise ValueError("unknown reaction: " + reaction)

    drug_enc = le_drug.transform([drugname])[0]
    reac_enc = le_reac.transform([reaction])[0]
    sex_enc = 0 if sex == "F" else 1
    drug_risk_rate = risk_rates["drug_risk"].get(drug_enc, 0.5)
    reac_risk_rate = risk_rates["reac_risk"].get(reac_enc, 0.5)
    combo_risk_rate = risk_rates["combo_risk"].get(f"{drug_enc}_{reac_enc}", 0.5)

    feature_names = [
        "drug",
        "reaction",
        "sex",
        "age",
        "drug_risk_rate",
        "reac_risk_rate",
        "combo_risk_rate",
    ]
    X = np.array(
        [
            [
                drug_enc,
                reac_enc,
                sex_enc,
                age,
                drug_risk_rate,
                reac_risk_rate,
                combo_risk_rate,
            ]
        ]
    )

    explainer = load_explainer()
    shap_values = explainer(X)
    sv_raw = shap_values.values
    sv = sv_raw[0, :, 1] if sv_raw.ndim == 3 else sv_raw[0]

    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]

    feature_display = {
        "drug": drugname,
        "reaction": reaction,
        "sex": sex,
        "age": age,
        "drug_risk_rate": round(drug_risk_rate, 3),
        "reac_risk_rate": round(reac_risk_rate, 3),
        "combo_risk_rate": round(combo_risk_rate, 3),
    }

    shap_result = [
        {
            "feature": name,
            "value": feature_display[name],
            "shap": round(float(sv[i]), 4),
        }
        for i, name in enumerate(feature_names)
    ]
    shap_result.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "drug": drugname,
        "reaction": reaction,
        "prediction": pred,
        "risk_label": "HIGH RISK" if pred == 1 else "LOW RISK",
        "probability": {
            "safe": round(float(prob[0]) * 100, 1),
            "risk": round(float(prob[1]) * 100, 1),
        },
        "shap": shap_result,
    }


# ── Routes ────────────────────────────────────


@analysis.route("/shap")
def shap_page():
    return render_template("shap.html")


@analysis.route("/api/shap")
def api_shap():
    drugname = request.args.get("drug", "").upper()
    reaction = request.args.get("reaction", "").upper()
    age = float(request.args.get("age", 50))
    sex = request.args.get("sex", "F")

    try:
        result = compute_shap(drugname, reaction, age, sex)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "model load failed: " + str(e)}), 500

    return jsonify(result)


@analysis.route("/api/shap/explain", methods=["POST"])
def api_shap_explain():
    """SHAP 기여도를 Ollama LLM이 한국어로 설명."""
    data = request.get_json() or {}
    drugname = data.get("drug", "").upper()
    reaction = data.get("reaction", "").upper()
    age = float(data.get("age", 50))
    sex = data.get("sex", "F")

    try:
        result = compute_shap(drugname, reaction, age, sex)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "model load failed: " + str(e)}), 500

    top_features = result["shap"][:3]
    lines = []
    for f in top_features:
        label = FEATURE_LABELS_KO.get(f["feature"], f["feature"])
        direction = "위험도를 높이는 방향" if f["shap"] >= 0 else "위험도를 낮추는 방향"
        lines.append(
            f"- {label} (값: {f['value']}) → {direction}으로 기여 (SHAP {f['shap']:+.3f})"
        )
    shap_lines = "\n".join(lines)

    prompt = f"""당신은 한국어 임상 데이터 설명 보조원입니다. 반드시 자연스러운 한국어와 약물명 등 필요한 영어 단어만 사용하세요.
절대 규칙:
- 중국어 한자, 힌디어, 일본어, 기타 외국어 문자를 절대 섞지 마세요.
- 아래 SHAP 분석 결과에 없는 의학적 주장을 새로 만들지 마세요.

[예측 결과]
약물: {result['drug']} / 부작용: {result['reaction']}
AI 판정: {result['risk_label']} (안전 {result['probability']['safe']}%, 위험 {result['probability']['risk']}%)

[SHAP 특성 기여도 Top 3 - 영향력 큰 순서]
{shap_lines}

[답변] 위 SHAP 기여도를 근거로, 어떤 요인이 이 예측에 가장 크게 영향을 줬는지 한국어 3문장 이내로 설명하세요. 숫자(SHAP 값)를 그대로 나열하지 말고 의미를 풀어서 설명하세요:"""

    try:
        response = http_requests.post(
            current_app.config.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.85},
            },
            timeout=60,
        )
        explanation = response.json().get("response", "설명 생성 실패")
        explanation = re.sub(
            r"[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]", "", explanation
        )
    except Exception as e:
        explanation = f"Ollama 오류: {str(e)}"

    return jsonify(
        {
            "drug": result["drug"],
            "reaction": result["reaction"],
            "top_features": [
                {
                    "feature": FEATURE_LABELS_KO.get(f["feature"], f["feature"]),
                    "value": f["value"],
                    "shap": f["shap"],
                }
                for f in top_features
            ],
            "explanation": explanation,
        }
    )


@analysis.route("/api/lime")
def api_lime():
    drugname = request.args.get("drug", "").upper()
    reaction = request.args.get("reaction", "").upper()
    age = float(request.args.get("age", 50))
    sex = request.args.get("sex", "F")

    try:
        from lime.lime_tabular import LimeTabularExplainer

        model, le_drug, le_reac = load_model()
        risk_rates = pickle.load(
            open(os.path.join(current_app.config["MODEL_DIR"], "risk_rates.pkl"), "rb")
        )

        if drugname not in le_drug.classes_:
            return jsonify({"error": "unknown drug: " + drugname}), 400
        if reaction not in le_reac.classes_:
            return jsonify({"error": "unknown reaction: " + reaction}), 400

        drug_enc = le_drug.transform([drugname])[0]
        reac_enc = le_reac.transform([reaction])[0]
        sex_enc = 0 if sex == "F" else 1
        drug_risk_rate = risk_rates["drug_risk"].get(drug_enc, 0.5)
        reac_risk_rate = risk_rates["reac_risk"].get(reac_enc, 0.5)
        combo_risk_rate = risk_rates["combo_risk"].get(f"{drug_enc}_{reac_enc}", 0.5)

        feature_names = [
            "drug",
            "reaction",
            "sex",
            "age",
            "drug_risk_rate",
            "reac_risk_rate",
            "combo_risk_rate",
        ]
        x = np.array(
            [
                [
                    drug_enc,
                    reac_enc,
                    sex_enc,
                    age,
                    drug_risk_rate,
                    reac_risk_rate,
                    combo_risk_rate,
                ]
            ]
        )

        np.random.seed(42)
        n_bg = 500
        bg = np.column_stack(
            [
                np.random.randint(0, max(len(le_drug.classes_), 1), n_bg),
                np.random.randint(0, max(len(le_reac.classes_), 1), n_bg),
                np.random.randint(0, 3, n_bg),
                np.random.uniform(0, 100, n_bg),
                np.random.uniform(0, 1, n_bg),
                np.random.uniform(0, 1, n_bg),
                np.random.uniform(0, 1, n_bg),
            ]
        )

        explainer = LimeTabularExplainer(
            bg,
            feature_names=feature_names,
            class_names=["low_risk", "high_risk"],
            mode="classification",
            random_state=42,
        )
        exp = explainer.explain_instance(
            x[0], model.predict_proba, num_features=7, num_samples=300, labels=(1,)
        )
        pred = int(model.predict(x)[0])
        prob = model.predict_proba(x)[0]

        lime_result = []
        for feat_expr, weight in exp.as_list(label=1):
            matched = next((f for f in feature_names if f in feat_expr), feat_expr)
            lime_result.append(
                {
                    "feature": matched,
                    "expression": feat_expr,
                    "weight": round(float(weight), 4),
                }
            )
        lime_result.sort(key=lambda x: abs(x["weight"]), reverse=True)

        return jsonify(
            {
                "drug": drugname,
                "reaction": reaction,
                "prediction": pred,
                "risk_label": "HIGH RISK" if pred == 1 else "LOW RISK",
                "probability": {
                    "safe": round(float(prob[0]) * 100, 1),
                    "risk": round(float(prob[1]) * 100, 1),
                },
                "lime": lime_result,
                "feature_values": {
                    "drug": drugname,
                    "reaction": reaction,
                    "sex": sex,
                    "age": age,
                    "drug_risk_rate": round(drug_risk_rate, 3),
                    "reac_risk_rate": round(reac_risk_rate, 3),
                    "combo_risk_rate": round(combo_risk_rate, 3),
                },
            }
        )

    except ImportError:
        return jsonify({"error": "lime not installed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
