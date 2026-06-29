"""
app/routes/analysis/forecast.py
Prophet 기반 시계열 예측 (다음 2개 분기 부작용 보고 수)
"""
import pandas as pd
from flask import jsonify, request

from . import analysis
from ._common import load_df


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
