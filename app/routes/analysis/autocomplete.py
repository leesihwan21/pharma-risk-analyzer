"""
app/routes/analysis/autocomplete.py
약물명 / 부작용명 자동완성
"""
from flask import jsonify, request

from . import analysis
from ._common import load_df


@analysis.route("/api/autocomplete")
def api_autocomplete():
    q = request.args.get("q", "").upper().strip()
    if len(q) < 2:
        return jsonify([])
    try:
        df = load_df()
        drugs = df["drugname"].str.upper().dropna().unique()
        matched = sorted([d for d in drugs if d.startswith(q)])[:10]
        return jsonify(matched)
    except Exception:
        return jsonify([])


@analysis.route("/api/autocomplete/reaction")
def api_autocomplete_reaction():
    q = request.args.get("q", "").upper().strip()
    if len(q) < 2:
        return jsonify([])
    try:
        df = load_df()
        reactions = df["pt"].str.upper().dropna().unique()
        matched = sorted([r for r in reactions if r.startswith(q)])[:10]
        return jsonify(matched)
    except Exception:
        return jsonify([])
