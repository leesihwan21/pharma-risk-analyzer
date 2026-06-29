"""
app/routes/analysis/prr.py
PRR 신호 탐지, EBGM, Emerging Signals, Favorites Alerts
"""

import math

from flask import jsonify, render_template
from app import cache
from app.models import FavoriteDrug
from .signals import compute_emerging_signals

from . import analysis
from ._common import load_df

# ── 공용 함수 ─────────────────────────────────


def compute_prr_summary(drugname: str, df=None):
    """PRR 신호 탐지 결과 반환 (없으면 None)."""
    if df is None:
        df = load_df()
    drugname = drugname.upper()

    drug_reports = df[df["drugname"].str.upper() == drugname]
    if len(drug_reports) == 0:
        return None

    other_reports = df[df["drugname"].str.upper() != drugname]
    total_drug = len(drug_reports)
    total_other = len(other_reports)

    if total_other == 0:
        return None

    top_reactions = drug_reports["pt"].value_counts().head(20).index.tolist()
    results = []

    for reac in top_reactions:
        a = len(drug_reports[drug_reports["pt"] == reac])
        b = total_drug
        c = len(other_reports[other_reports["pt"] == reac])
        d = total_other

        if c == 0 or b == 0:
            continue

        prr = (a / b) / (c / d)
        try:
            se = math.sqrt((1 / a) - (1 / b) + (1 / c) - (1 / d))
            prr_lower = math.exp(math.log(prr) - 1.96 * se)
            prr_upper = math.exp(math.log(prr) + 1.96 * se)
        except (ValueError, ZeroDivisionError):
            prr_lower = prr_upper = prr

        is_signal = prr >= 2 and a >= 3
        results.append(
            {
                "reaction": reac,
                "drug_count": int(a),
                "drug_total": int(b),
                "other_count": int(c),
                "other_total": int(d),
                "drug_pct": round(a / b * 100, 2),
                "other_pct": round(c / d * 100, 2),
                "prr": round(prr, 2),
                "prr_lower": round(prr_lower, 2),
                "prr_upper": round(prr_upper, 2),
                "is_signal": is_signal,
                "signal_level": (
                    "🔴 강한 신호"
                    if prr >= 5 and a >= 3
                    else "🟡 신호" if prr >= 2 and a >= 3 else "⚪ 비신호"
                ),
            }
        )

    results.sort(key=lambda x: x["prr"], reverse=True)
    signal_count = sum(1 for r in results if r["is_signal"])
    strong_signal_count = sum(
        1 for r in results if r["prr"] >= 5 and r["drug_count"] >= 3
    )

    return {
        "drugname": drugname,
        "total_reports": total_drug,
        "signal_count": signal_count,
        "strong_signal_count": strong_signal_count,
        "results": results,
    }


def compute_emerging_signals(
    drugname: str, df=None, min_count: int = 3, prr_threshold: float = 2.0
):
    """최신 분기 vs 이전 분기 신규 신호 탐지."""
    if df is None:
        df = load_df()
    drugname = drugname.upper()

    if "quarter" not in df.columns:
        return None

    quarters = sorted(df["quarter"].dropna().unique())
    if len(quarters) < 2:
        return None
    latest_q = quarters[-1]

    df_latest = df[df["quarter"] == latest_q]
    df_history = df[df["quarter"] != latest_q]

    drug_latest = df_latest[df_latest["drugname"].str.upper() == drugname]
    if len(drug_latest) == 0:
        return None

    other_latest = df_latest[df_latest["drugname"].str.upper() != drugname]
    drug_history = df_history[df_history["drugname"].str.upper() == drugname]
    other_history = df_history[df_history["drugname"].str.upper() != drugname]

    b = len(drug_latest)
    d = len(other_latest)
    b2 = len(drug_history)
    d2 = len(other_history)
    if b == 0 or d == 0:
        return None

    def _prr(a, b, c, d):
        if a == 0 or c == 0 or b == 0 or d == 0:
            return 0.0
        return (a / b) / (c / d)

    emerging = []
    for reac in drug_latest["pt"].value_counts().index.tolist():
        a = len(drug_latest[drug_latest["pt"] == reac])
        c = len(other_latest[other_latest["pt"] == reac])
        prr_latest = _prr(a, b, c, d)
        if not (prr_latest >= prr_threshold and a >= min_count):
            continue

        a2 = len(drug_history[drug_history["pt"] == reac]) if b2 > 0 else 0
        c2 = len(other_history[other_history["pt"] == reac]) if d2 > 0 else 0
        prr_history = _prr(a2, b2, c2, d2)

        if not (prr_history >= prr_threshold and a2 >= min_count):
            emerging.append(
                {
                    "reaction": reac,
                    "prr_latest": round(prr_latest, 2),
                    "prr_history": round(prr_history, 2),
                    "latest_count": int(a),
                    "quarter": str(latest_q),
                }
            )

    return {"drugname": drugname, "latest_quarter": str(latest_q), "emerging": emerging}


# ── Routes ────────────────────────────────────


@analysis.route("/prr")
def prr_page():
    return render_template("prr.html")


@analysis.route("/api/prr/<drugname>")
@cache.cached(timeout=600)
def calculate_prr(drugname: str):
    summary = compute_prr_summary(drugname)
    if summary is None:
        return jsonify({"error": f"약물을 찾을 수 없어요: {drugname.upper()}"}), 404
    return jsonify(summary)


@analysis.route("/api/favorites/alerts")
def favorites_alerts():
    """즐겨찾기 약물의 PRR + 신규 신호 알림."""
    from app.models import FavoriteDrug
    favorites = FavoriteDrug.query.all()
    if not favorites:
        return jsonify({"alerts": []})

    df = load_df()
    alerts = []
    seen = set()

    for fav in favorites:
        drugname = fav.drugname.upper()
        if drugname in seen:
            continue
        seen.add(drugname)

        summary = compute_prr_summary(drugname, df=df)
        if summary is None:
            continue

        top_signals = [r for r in summary["results"] if r["is_signal"]][:3]
        if summary["strong_signal_count"] > 0:
            level = "strong"
        elif summary["signal_count"] > 0:
            level = "signal"
        else:
            level = "none"

        emerging_result = compute_emerging_signals(drugname, df=df)
        emerging = emerging_result["emerging"] if emerging_result else []
        if emerging:
            level = "new"

        alerts.append({
            "drugname": summary["drugname"],
            "signal_count": summary["signal_count"],
            "strong_signal_count": summary["strong_signal_count"],
            "level": level,
            "top_signals": [
                {"reaction": r["reaction"], "prr": r["prr"], "signal_level": r["signal_level"]}
                for r in top_signals
            ],
            "emerging_signals": [
                {"reaction": e["reaction"], "prr_latest": e["prr_latest"], "quarter": e["quarter"]}
                for e in emerging[:3]
            ]
        })

    level_order = {"new": 0, "strong": 1, "signal": 2, "none": 3}
    alerts.sort(key=lambda a: level_order[a["level"]])
    return jsonify({"alerts": alerts})
