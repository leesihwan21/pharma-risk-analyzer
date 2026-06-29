"""
app/routes/analysis/trend.py
Trend 조회 (단순 분기별 보고 건수)
"""
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
