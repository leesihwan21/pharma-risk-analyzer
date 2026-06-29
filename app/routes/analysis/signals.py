"""
app/routes/analysis/signals.py
신호 탐지: Emerging Signals, 분기별 PRR 급변 탐지
"""
import math

from flask import jsonify, request
from app import cache

from . import analysis
from ._common import load_df


def compute_emerging_signals(drugname: str, df=None, min_count: int = 3, prr_threshold: float = 2.0):
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

    df_latest  = df[df["quarter"] == latest_q]
    df_history = df[df["quarter"] != latest_q]

    drug_latest = df_latest[df_latest["drugname"].str.upper() == drugname]
    if len(drug_latest) == 0:
        return None

    other_latest  = df_latest[df_latest["drugname"].str.upper() != drugname]
    drug_history  = df_history[df_history["drugname"].str.upper() == drugname]
    other_history = df_history[df_history["drugname"].str.upper() != drugname]

    b  = len(drug_latest)
    d  = len(other_latest)
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


@analysis.route("/api/signals/emerging/<drugname>")
@cache.cached(timeout=600)
def api_emerging_signals(drugname: str):
    result = compute_emerging_signals(drugname)
    if result is None:
        return jsonify({"error": f"분석할 수 없습니다: {drugname.upper()} (데이터 또는 분기 정보 부족)"}), 404
    return jsonify(result)


@analysis.route("/api/signals/quarterly_trend/<drugname>")
def api_quarterly_prr_trend(drugname: str):
    """분기별 PRR 변화 추이 + 급변(50% 이상 상승) 플래그."""
    try:
        df = load_df()
        drugname = drugname.upper()

        if "quarter" not in df.columns:
            return jsonify({"error": "quarter column not found"}), 400

        drug_df = df[df["drugname"].str.upper() == drugname]
        if drug_df.empty:
            return jsonify({"error": f"Drug not found: {drugname}"}), 404

        quarters = sorted(df["quarter"].dropna().unique())
        if len(quarters) < 2:
            return jsonify({"error": "Not enough quarters (need 2+)"}), 400

        results = []
        for q in quarters:
            df_q   = df[df["quarter"] == q]
            drug_q = df_q[df_q["drugname"].str.upper() == drugname]
            other_q = df_q[df_q["drugname"].str.upper() != drugname]
            b, d = len(drug_q), len(other_q)
            if b == 0 or d == 0:
                continue

            reac_data = []
            for reac in drug_q["pt"].value_counts().head(20).index:
                a = len(drug_q[drug_q["pt"] == reac])
                c = len(other_q[other_q["pt"] == reac])
                if c == 0:
                    continue
                prr = round((a / b) / (c / d), 2)
                if prr >= 2.0 and a >= 3:
                    reac_data.append({"reaction": reac, "prr": prr, "count": int(a)})

            results.append(
                {
                    "quarter": q,
                    "total_reports": int(b),
                    "signals": sorted(reac_data, key=lambda x: x["prr"], reverse=True)[:10],
                }
            )

        spike_alerts = []
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            prev_prr_map = {s["reaction"]: s["prr"] for s in prev["signals"]}

            for sig in curr["signals"]:
                reac     = sig["reaction"]
                curr_prr = sig["prr"]
                prev_prr = prev_prr_map.get(reac, 0)

                if prev_prr == 0:
                    spike_alerts.append(
                        {
                            "reaction": reac, "type": "NEW_SIGNAL",
                            "quarter": curr["quarter"], "prev_quarter": prev["quarter"],
                            "curr_prr": curr_prr, "prev_prr": prev_prr,
                            "change_pct": None, "count": sig["count"],
                        }
                    )
                elif curr_prr >= prev_prr * 1.5:
                    spike_alerts.append(
                        {
                            "reaction": reac, "type": "SPIKE",
                            "quarter": curr["quarter"], "prev_quarter": prev["quarter"],
                            "curr_prr": curr_prr, "prev_prr": prev_prr,
                            "change_pct": round((curr_prr - prev_prr) / prev_prr * 100, 1),
                            "count": sig["count"],
                        }
                    )

        all_reactions = {s["reaction"] for r in results for s in r["signals"]}
        trend_by_reaction = {}
        for reac in list(all_reactions)[:8]:
            trend_by_reaction[reac] = [
                {
                    "quarter": r["quarter"],
                    "prr": {s["reaction"]: s["prr"] for s in r["signals"]}.get(reac, 0),
                }
                for r in results
            ]

        return jsonify(
            {
                "drug": drugname,
                "quarters": [r["quarter"] for r in results],
                "quarterly_data": results,
                "spike_alerts": spike_alerts,
                "trend_by_reaction": trend_by_reaction,
                "total_spikes": len(spike_alerts),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
