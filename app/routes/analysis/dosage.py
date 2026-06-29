"""
app/routes/analysis/dosage.py
신장기능(CrCl), 소아용량, BSA 계산
"""

import math

from flask import jsonify, render_template, request

from . import analysis


@analysis.route("/dosage")
def dosage_page():
    return render_template("dosage.html")


@analysis.route("/api/dosage/crcl", methods=["POST"])
def api_crcl():
    data = request.get_json()
    age = float(data.get("age", 0))
    weight = float(data.get("weight", 0))
    creatinine = float(data.get("creatinine", 0))
    sex = data.get("sex", "M")

    if not all([age, weight, creatinine]):
        return jsonify({"error": "missing values"}), 400

    crcl = ((140 - age) * weight) / (72 * creatinine)
    if sex == "F":
        crcl *= 0.85

    if crcl >= 90:
        stage, dose_adj, color = "Normal (G1)", "No dose adjustment required", "green"
    elif crcl >= 60:
        stage, dose_adj, color = (
            "Mild reduction (G2)",
            "Dose adjustment may be required",
            "yellow",
        )
    elif crcl >= 30:
        stage, dose_adj, color = (
            "Moderate reduction (G3)",
            "Reduce dose by 50-75%",
            "orange",
        )
    elif crcl >= 15:
        stage, dose_adj, color = "Severe reduction (G4)", "Reduce dose by 25-50%", "red"
    else:
        stage, dose_adj, color = (
            "Renal failure (G5)",
            "Nephrotoxic drugs contraindicated",
            "darkred",
        )

    return jsonify(
        {"crcl": round(crcl, 1), "stage": stage, "dose_adj": dose_adj, "color": color}
    )


@analysis.route("/api/dosage/pediatric", methods=["POST"])
def api_pediatric():
    data = request.get_json()
    adult_dose = float(data.get("adult_dose", 0))
    age = float(data.get("age", 0))
    weight = float(data.get("weight", 0))
    height = float(data.get("height", 0))

    if not adult_dose:
        return jsonify({"error": "missing values"}), 400

    results = {}
    if weight:
        results["clark"] = round(adult_dose * weight / 70, 2)
    if age:
        results["young"] = round(adult_dose * age / (age + 12), 2)
    if weight and height:
        bsa = math.sqrt((height * weight) / 3600)
        results["bsa"] = round(adult_dose * bsa / 1.73, 2)
        results["bsa_value"] = round(bsa, 2)

    return jsonify(results)


@analysis.route("/api/dosage/bsa", methods=["POST"])
def api_bsa():
    data = request.get_json()
    weight = float(data.get("weight", 0))
    height = float(data.get("height", 0))
    dose_per_m2 = float(data.get("dose_per_m2", 0))

    if not all([weight, height]):
        return jsonify({"error": "missing values"}), 400

    bsa = math.sqrt((height * weight) / 3600)
    total_dose = round(bsa * dose_per_m2, 2) if dose_per_m2 else None
    bsa_dubois = 0.007184 * (height**0.725) * (weight**0.425)

    return jsonify(
        {
            "bsa_mosteller": round(bsa, 3),
            "bsa_dubois": round(bsa_dubois, 3),
            "total_dose": total_dose,
        }
    )
