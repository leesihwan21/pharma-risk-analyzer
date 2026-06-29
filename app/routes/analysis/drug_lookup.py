"""
app/routes/analysis/drug_lookup.py
Drug Lookup (MFDS/OpenFDA), Drug Shape, Drug Vision (YOLOv8+Claude),
MedDRA SOC 분석, Autocomplete
"""

import base64

import requests as http_requests

from flask import jsonify, render_template, request, current_app
from app import cache

from . import analysis
from ._common import load_df

# ── MedDRA SOC 매핑 ───────────────────────────

MEDDRA_SOC_MAP = {
    "FATIGUE": "General disorders",
    "MALAISE": "General disorders",
    "PYREXIA": "General disorders",
    "PAIN": "General disorders",
    "SWELLING": "General disorders",
    "OEDEMA": "General disorders",
    "GENERAL PHYSICAL HEALTH DETERIORATION": "General disorders",
    "ASTHENIA": "General disorders",
    "CHILLS": "General disorders",
    "NAUSEA": "Gastrointestinal disorders",
    "VOMITING": "Gastrointestinal disorders",
    "DIARRHOEA": "Gastrointestinal disorders",
    "ABDOMINAL DISCOMFORT": "Gastrointestinal disorders",
    "ABDOMINAL PAIN": "Gastrointestinal disorders",
    "CONSTIPATION": "Gastrointestinal disorders",
    "DYSPEPSIA": "Gastrointestinal disorders",
    "STOMATITIS": "Gastrointestinal disorders",
    "ARTHRALGIA": "Musculoskeletal disorders",
    "ARTHROPATHY": "Musculoskeletal disorders",
    "JOINT SWELLING": "Musculoskeletal disorders",
    "MOBILITY DECREASED": "Musculoskeletal disorders",
    "MYALGIA": "Musculoskeletal disorders",
    "BACK PAIN": "Musculoskeletal disorders",
    "BONE PAIN": "Musculoskeletal disorders",
    "HEADACHE": "Nervous system disorders",
    "DIZZINESS": "Nervous system disorders",
    "PARAESTHESIA": "Nervous system disorders",
    "NEUROPATHY PERIPHERAL": "Nervous system disorders",
    "TREMOR": "Nervous system disorders",
    "SOMNOLENCE": "Nervous system disorders",
    "DYSPNOEA": "Respiratory disorders",
    "COUGH": "Respiratory disorders",
    "PNEUMONIA": "Respiratory disorders",
    "PULMONARY EMBOLISM": "Respiratory disorders",
    "RHINORRHOEA": "Respiratory disorders",
    "RASH": "Skin disorders",
    "PRURITUS": "Skin disorders",
    "ALOPECIA": "Skin disorders",
    "URTICARIA": "Skin disorders",
    "ERYTHEMA": "Skin disorders",
    "DERMATITIS": "Skin disorders",
    "HEPATIC ENZYME INCREASED": "Investigations",
    "BLOOD CHOLESTEROL INCREASED": "Investigations",
    "WEIGHT INCREASED": "Investigations",
    "WEIGHT DECREASED": "Investigations",
    "ALANINE AMINOTRANSFERASE INCREASED": "Investigations",
    "BLOOD CREATININE INCREASED": "Investigations",
    "DRUG HYPERSENSITIVITY": "Immune system disorders",
    "HYPERSENSITIVITY": "Immune system disorders",
    "ANAPHYLACTIC REACTION": "Immune system disorders",
    "HYPERTENSION": "Vascular disorders",
    "HYPOTENSION": "Vascular disorders",
    "FLUSHING": "Vascular disorders",
    "DEEP VEIN THROMBOSIS": "Vascular disorders",
    "URINARY TRACT INFECTION": "Infections and infestations",
    "NASOPHARYNGITIS": "Infections and infestations",
    "UPPER RESPIRATORY TRACT INFECTION": "Infections and infestations",
    "RHEUMATOID ARTHRITIS": "Musculoskeletal disorders",
    "SYSTEMIC LUPUS ERYTHEMATOSUS": "Immune system disorders",
    "OFF LABEL USE": "Social circumstances",
    "DRUG INEFFECTIVE": "General disorders",
    "DRUG INTOLERANCE": "General disorders",
    "CONDITION AGGRAVATED": "General disorders",
    "PRODUCT USE IN UNAPPROVED INDICATION": "Social circumstances",
    "INFUSION RELATED REACTION": "General disorders",
}


# ── Routes ────────────────────────────────────


@analysis.route("/drug-lookup")
def drug_lookup_page():
    return render_template("drug_lookup.html")


@analysis.route("/api/drug-lookup")
def api_drug_lookup():
    drugname = request.args.get("name", "").strip()
    if not drugname:
        return jsonify({"error": "no name"}), 400

    api_key = current_app.config.get("MFDS_API_KEY", "")
    results = {"korean": None, "openfda": None}

    try:
        mfds_url = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
        r = http_requests.get(
            mfds_url,
            params={
                "serviceKey": api_key,
                "item_name": drugname,
                "type": "json",
                "numOfRows": 1,
            },
            timeout=5,
        )
        items = r.json().get("body", {}).get("items", [])
        if items:
            item = items[0]
            results["korean"] = {
                "name": item.get("ITEM_NAME", ""),
                "company": item.get("ENTP_NAME", ""),
                "shape": item.get("DRUG_SHAPE", ""),
                "color": item.get("COLOR_CLASS1", ""),
                "etc_otc": item.get("ETC_OTC_CODE", ""),
                "class_name": item.get("CLASS_NAME", ""),
                "img_url": item.get("ITEM_IMAGE", ""),
            }
    except Exception as e:
        results["korean_error"] = str(e)

    try:
        r = http_requests.get(
            "https://api.fda.gov/drug/label.json",
            params={
                "search": f'openfda.brand_name:"{drugname.upper()}" OR openfda.generic_name:"{drugname.upper()}"',
                "limit": 1,
            },
            timeout=5,
        )
        items = r.json().get("results", [])
        if items:
            item = items[0]
            openfda = item.get("openfda", {})
            results["openfda"] = {
                "brand_name": (
                    openfda.get("brand_name", [""])[0]
                    if openfda.get("brand_name")
                    else ""
                ),
                "generic_name": (
                    openfda.get("generic_name", [""])[0]
                    if openfda.get("generic_name")
                    else ""
                ),
                "manufacturer": (
                    openfda.get("manufacturer_name", [""])[0]
                    if openfda.get("manufacturer_name")
                    else ""
                ),
                "purpose": (
                    item.get("purpose", [""])[0][:300] if item.get("purpose") else ""
                ),
                "warnings": (
                    item.get("warnings", [""])[0][:300] if item.get("warnings") else ""
                ),
                "adverse_reactions": (
                    item.get("adverse_reactions", [""])[0][:500]
                    if item.get("adverse_reactions")
                    else ""
                ),
                "dosage": (
                    item.get("dosage_and_administration", [""])[0][:300]
                    if item.get("dosage_and_administration")
                    else ""
                ),
            }
    except Exception as e:
        results["openfda_error"] = str(e)

    if not results["korean"] and not results["openfda"]:
        return jsonify({"error": "not found", "drug": drugname}), 404

    results["drug"] = drugname
    return jsonify(results)


@analysis.route("/api/drug-shape")
def api_drug_shape():
    shape = request.args.get("shape", "")
    color = request.args.get("color", "")
    front = request.args.get("front", "")
    back = request.args.get("back", "")
    api_key = current_app.config.get("MFDS_API_KEY", "")

    try:
        url = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
        params = {"serviceKey": api_key, "type": "json", "numOfRows": 10}
        if shape:
            params["drug_shape"] = shape
        if color:
            params["color_class1"] = color
        if front:
            params["print_front"] = front
        if back:
            params["print_back"] = back

        r = http_requests.get(url, params=params, timeout=5)
        data = r.json()
        items = data.get("body", {}).get("items", [])
        result = [
            {
                "name": item.get("ITEM_NAME", ""),
                "company": item.get("ENTP_NAME", ""),
                "shape": item.get("DRUG_SHAPE", ""),
                "color": item.get("COLOR_CLASS1", ""),
                "etc_otc": item.get("ETC_OTC_CODE", ""),
                "class_name": item.get("CLASS_NAME", ""),
                "chart": item.get("CHART", ""),
                "img_url": item.get("ITEM_IMAGE", ""),
                "print_front": item.get("PRINT_FRONT", ""),
                "print_back": item.get("PRINT_BACK", ""),
            }
            for item in items
        ]
        return jsonify(
            {"items": result, "total": data.get("body", {}).get("totalCount", 0)}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@analysis.route("/api/drug-vision", methods=["POST"])
def api_drug_vision():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    image_file = request.files["image"]
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
    media_type = image_file.content_type or "image/jpeg"
    api_key = current_app.config.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return jsonify({"error": "no api key"}), 500

    try:
        payload = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "system": [
                {
                    "type": "text",
                    "text": (
                        "당신은 전문 의약품 식별 및 알약 인식 AI 어시스턴트입니다. "
                        "입력된 이미지의 형태, 색상, 각인, 제형(정제, 캡슐 등)을 기반으로 대한민국 식약처 및 OpenFDA 기준에 부합하는 정확한 의약품명을 추론해야 합니다. "
                        "반드시 부가적인 설명 없이 '정확한 약물명 혹은 성분명'만 단답형으로 출력하세요. "
                        "만약 약물이 명확히 보이지 않거나 식별이 불가능하다면 오직 '알 수 없음'이라고만 답변하여 할루시네이션(거짓 정보)을 방지하십시오."
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "이 이미지에서 약물/의약품을 식별하여 약물명만 간단하게 답해주세요.",
                        },
                    ],
                }
            ],
        }
        r = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        result = r.json()
        if "usage" in result:
            print(
                f"[Anthropic Usage] input={result['usage'].get('input_tokens')} "
                f"output={result['usage'].get('output_tokens')} "
                f"cache_create={result['usage'].get('cache_creation_input_tokens', 0)} "
                f"cache_read={result['usage'].get('cache_read_input_tokens', 0)}"
            )
        drug_name = result["content"][0]["text"].strip()
    except Exception as e:
        return jsonify({"error": "vision failed: " + str(e)}), 500

    if drug_name == "알 수 없음" or not drug_name:
        return jsonify({"error": "cannot identify drug", "raw": drug_name}), 404
    return jsonify({"detected_drug": drug_name})


@analysis.route("/api/soc/<drugname>")
@cache.cached(timeout=600)
def soc_analysis(drugname: str):
    df = load_df()
    drugname = drugname.upper()
    result = df[df["drugname"].str.upper() == drugname]

    if len(result) == 0:
        return jsonify({"error": f"약물을 찾을 수 없어요: {drugname}"}), 404

    top_reac = result["pt"].value_counts().head(50)
    soc_counts = {}
    mapped_reactions = []

    for reac, cnt in top_reac.items():
        soc = MEDDRA_SOC_MAP.get(reac.upper(), "Other")
        soc_counts[soc] = soc_counts.get(soc, 0) + cnt
        mapped_reactions.append(
            {
                "pt": reac,
                "count": int(cnt),
                "pct": round(cnt / len(result) * 100, 2),
                "soc": soc,
            }
        )

    soc_summary = sorted(
        [
            {"soc": k, "count": v, "pct": round(v / len(result) * 100, 1)}
            for k, v in soc_counts.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    return jsonify(
        {
            "drugname": drugname,
            "total_reports": len(result),
            "soc_summary": soc_summary,
            "reactions": mapped_reactions,
            "mapped_count": sum(1 for r in mapped_reactions if r["soc"] != "Other"),
            "note": "MedDRA SOC 매핑 (빈출 PT 기준 수동 매핑, 포트폴리오용)",
        }
    )


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
