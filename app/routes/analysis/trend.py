"""
app/routes/analysis/trend.py
Trend 조회, Prophet 시계열 예측, 분기별 PRR 급변 탐지
"""

import pandas as pd

from flask import jsonify, render_template, request

from . import analysis
from ._common import load_df


@analysis.route("/trend")
def trend_page():
    return render_template("trend.html")


@analysis.route("/api/trend")
def api_trend():
    drugname = request.args.get("drug", "").upper()
    if not drugname:
        return jsonify({"error": "no drug"}), 400

    df = load_df()
    filtered = df[df["drugname"] == drugname]
    if filtered.empty:
        return jsonify({"error": "not found"}), 404

    trend = (
        filtered.groupby("quarter")
        .size()
        .reset_index(name="count")
        .sort_values("quarter")
    )
    return jsonify({"drug": drugname, "trend": trend.to_dict(orient="records")})


@analysis.route("/api/trend/forecast")
def api_trend_forecast():
    """Prophet으로 다음 2개 분기 부작용 보고 수 예측."""
    drugname = request.args.get("drug", "").upper()
    if not drugname:
        return jsonify({"error": "no drug"}), 400

    try:
        from prophet import Prophet

        df = load_df()
        filtered = df[df["drugname"] == drugname]
        if filtered.empty:
            return jsonify({"error": f"Drug not found: {drugname}"}), 404

        trend = (
            filtered.groupby("quarter")
            .size()
            .reset_index(name="count")
            .sort_values("quarter")
        )
        if len(trend) < 3:
            return jsonify({"error": "Not enough quarterly data (need 3+)"}), 400

        def quarter_to_date(q: str) -> str:
            year, qn = q[:4], q[5]
            month = {"1": "01", "2": "04", "3": "07", "4": "10"}[qn]
            return f"{year}-{month}-01"

        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(trend["quarter"].apply(quarter_to_date)),
                "y": trend["count"].astype(float),
            }
        )

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.8,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=2, freq="QS")
        forecast = model.predict(future)

        hist_quarters = set(trend["quarter"].tolist())

        def date_to_quarter(dt) -> str:
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year}q{q}"

        result_rows = []
        for _, row in forecast.iterrows():
            q = date_to_quarter(row["ds"])
            is_forecast = q not in hist_quarters
            result_rows.append(
                {
                    "quarter": q,
                    "yhat": max(0, round(float(row["yhat"]), 1)),
                    "yhat_lower": max(0, round(float(row["yhat_lower"]), 1)),
                    "yhat_upper": max(0, round(float(row["yhat_upper"]), 1)),
                    "is_forecast": is_forecast,
                }
            )

        actual_map = dict(zip(trend["quarter"], trend["count"]))
        for r in result_rows:
            r["actual"] = (
                int(actual_map[r["quarter"]]) if r["quarter"] in actual_map else None
            )

        return jsonify(
            {
                "drug": drugname,
                "forecast": result_rows,
                "quarters": len(trend),
                "forecast_periods": 2,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            df_q = df[df["quarter"] == q]
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
                    "signals": sorted(reac_data, key=lambda x: x["prr"], reverse=True)[
                        :10
                    ],
                }
            )

        spike_alerts = []
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            prev_prr_map = {s["reaction"]: s["prr"] for s in prev["signals"]}

            for sig in curr["signals"]:
                reac = sig["reaction"]
                curr_prr = sig["prr"]
                prev_prr = prev_prr_map.get(reac, 0)

                if prev_prr == 0:
                    spike_alerts.append(
                        {
                            "reaction": reac,
                            "type": "NEW_SIGNAL",
                            "quarter": curr["quarter"],
                            "prev_quarter": prev["quarter"],
                            "curr_prr": curr_prr,
                            "prev_prr": prev_prr,
                            "change_pct": None,
                            "count": sig["count"],
                        }
                    )
                elif curr_prr >= prev_prr * 1.5:
                    spike_alerts.append(
                        {
                            "reaction": reac,
                            "type": "SPIKE",
                            "quarter": curr["quarter"],
                            "prev_quarter": prev["quarter"],
                            "curr_prr": curr_prr,
                            "prev_prr": prev_prr,
                            "change_pct": round(
                                (curr_prr - prev_prr) / prev_prr * 100, 1
                            ),
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
