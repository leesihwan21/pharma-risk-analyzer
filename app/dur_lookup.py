# app/dur_lookup.py
"""
FAERS(미국 FDA) 약물명을 한국 식약처 DUR(의약품안전사용서비스) 병용금기 조회에
연결하기 위한 매핑 테이블 및 조회 함수.

주의: DUR 병용금기 API는 한글 품목명/성분명 기준으로 조회되므로,
FAERS의 영문 약물명과 직접 매칭이 불가능합니다. 따라서 본 프로젝트의
FAERS 데이터셋(류마티스 관절염/자가면역질환 치료제 중심)에서 빈도가 높은
약물 위주로 한글 성분명을 수동 매핑하였습니다.

생물학적제제(바이오의약품)는 대부분 경구약 중심의 DUR 병용금기 DB에
등재되어 있지 않으므로, 조회 시 "DUR 병용금기 데이터베이스 미등재"로
명확히 표시합니다 (이 역시 유의미한 분석 결과입니다).
"""

import requests

DUR_BASE_URL = (
    "https://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03"
)

# FAERS 영문 약물명 -> (한글 검색명, 영문 성분명, 비고)
# 비고가 'biologic'인 약물은 DUR 병용금기 DB 대상이 아닐 가능성이 높음(소분자 경구약 위주 DB)
FAERS_TO_KOREAN_DUR = {
    "METHOTREXATE": {
        "kr_name": "메토트렉세이트",
        "ingr_eng": "METHOTREXATE",
        "note": "small_molecule",
    },
    "SULFASALAZINE": {
        "kr_name": "설파살라진",
        "ingr_eng": "SULFASALAZINE",
        "note": "small_molecule",
    },
    "PREDNISONE": {
        "kr_name": "프레드니손",
        "ingr_eng": "PREDNISONE",
        "note": "small_molecule",
    },
    "FOLIC ACID": {
        "kr_name": "폴산",
        "ingr_eng": "FOLIC ACID",
        "note": "small_molecule",
    },
    "HYDROXYCHLOROQUINE": {
        "kr_name": "히드록시클로로퀸",
        "ingr_eng": "HYDROXYCHLOROQUINE",
        "note": "small_molecule",
    },
    "HYDROXYCHLOROQUINE SULFATE": {
        "kr_name": "히드록시클로로퀸",
        "ingr_eng": "HYDROXYCHLOROQUINE",
        "note": "small_molecule",
    },
    "LEFLUNOMIDE": {
        "kr_name": "레플루노미드",
        "ingr_eng": "LEFLUNOMIDE",
        "note": "small_molecule",
    },
    "ARAVA": {
        "kr_name": "레플루노미드",
        "ingr_eng": "LEFLUNOMIDE",
        "note": "small_molecule",
    },
    "ACETAMINOPHEN": {
        "kr_name": "아세트아미노펜",
        "ingr_eng": "ACETAMINOPHEN",
        "note": "small_molecule",
    },
    "DICLOFENAC SODIUM": {
        "kr_name": "디클로페낙나트륨",
        "ingr_eng": "DICLOFENAC",
        "note": "small_molecule",
    },
    "FOSAMAX": {
        "kr_name": "알렌드론산",
        "ingr_eng": "ALENDRONATE",
        "note": "small_molecule",
    },
    "CETIRIZINE": {
        "kr_name": "세티리진",
        "ingr_eng": "CETIRIZINE",
        "note": "small_molecule",
    },
    "CETIRIZINE HYDROCHLORIDE": {
        "kr_name": "세티리진",
        "ingr_eng": "CETIRIZINE",
        "note": "small_molecule",
    },
    "CORTISONE ACETATE": {
        "kr_name": "코르티손아세테이트",
        "ingr_eng": "CORTISONE",
        "note": "small_molecule",
    },
    "ASPIRIN": {"kr_name": "아스피린", "ingr_eng": "ASPIRIN", "note": "small_molecule"},
    # 생물학적제제 (DUR 병용금기 DB 비대상 가능성 높음 - 참고용으로만 매핑)
    "ACTEMRA": {"kr_name": "토실리주맙", "ingr_eng": "TOCILIZUMAB", "note": "biologic"},
    "ORENCIA": {"kr_name": "아바타셉트", "ingr_eng": "ABATACEPT", "note": "biologic"},
    "ENBREL": {"kr_name": "에타너셉트", "ingr_eng": "ETANERCEPT", "note": "biologic"},
    "XELJANZ": {"kr_name": "토파시티닙", "ingr_eng": "TOFACITINIB", "note": "biologic"},
    "HUMIRA": {"kr_name": "아달리무맙", "ingr_eng": "ADALIMUMAB", "note": "biologic"},
    "ADALIMUMAB": {
        "kr_name": "아달리무맙",
        "ingr_eng": "ADALIMUMAB",
        "note": "biologic",
    },
    "CIMZIA": {"kr_name": "서톨리주맙", "ingr_eng": "CERTOLIZUMAB", "note": "biologic"},
    "REMICADE": {"kr_name": "인플릭시맙", "ingr_eng": "INFLIXIMAB", "note": "biologic"},
    "INFLECTRA": {
        "kr_name": "인플릭시맙",
        "ingr_eng": "INFLIXIMAB",
        "note": "biologic",
    },
    "INFLIXIMAB": {
        "kr_name": "인플릭시맙",
        "ingr_eng": "INFLIXIMAB",
        "note": "biologic",
    },
    "SIMPONI": {"kr_name": "골리무맙", "ingr_eng": "GOLIMUMAB", "note": "biologic"},
    "COSENTYX": {
        "kr_name": "세쿠키누맙",
        "ingr_eng": "SECUKINUMAB",
        "note": "biologic",
    },
    "RITUXIMAB": {"kr_name": "리툭시맙", "ingr_eng": "RITUXIMAB", "note": "biologic"},
}


def get_dur_mapping(drugname: str):
    """FAERS 약물명을 DUR 매핑 정보로 변환. 매핑이 없으면 None."""
    return FAERS_TO_KOREAN_DUR.get(drugname.upper())


def query_dur_taboo(kr_name: str, api_key: str, timeout: int = 10):
    """
    DUR 병용금기 정보조회 (getUsjntTabooInfoList03).
    kr_name: 한글 성분명/품목명 검색어 (itemName 파라미터)
    반환: 병용금기 대상 항목 리스트 (items), 오류/결과없음 시 빈 리스트
    """
    params = {
        "serviceKey": api_key,
        "itemName": kr_name,
        "type": "json",
        "numOfRows": 500,
        "pageNo": 1,
    }
    try:
        res = requests.get(DUR_BASE_URL, params=params, timeout=timeout)
        data = res.json()
        header = data.get("header", {})
        if header.get("resultCode") not in (None, "00", "0"):
            print(
                f"[DUR API] {kr_name} resultCode={header.get('resultCode')} msg={header.get('resultMsg')}"
            )
        body = data.get("body", {})
        items = body.get("items", [])

        if not items:
            return []

        # 공공데이터포털 API 특성상 items가 곧바로 list로 오거나,
        # {"item": {...}} / {"item": [...]} 형태로 올 수 있어 모두 처리
        if isinstance(items, list):
            return items
        if isinstance(items, dict):
            item = items.get("item", [])
            if isinstance(item, dict):
                return [item]
            if isinstance(item, list):
                return item
        return []
    except Exception as e:
        print(f"[DUR API ERROR] {kr_name}: {e}")
        return []


def check_combo_dur_taboo(drug_a: str, drug_b: str, api_key: str):
    """
    두 약물(FAERS 영문명) 조합이 한국 DUR 병용금기 기준에 해당하는지 확인.

    반환 예시:
    {
        'checked': True,
        'is_taboo': True/False,
        'drug_a_mapped': {...} / None,
        'drug_b_mapped': {...} / None,
        'reason': '설명 텍스트'
    }
    """
    map_a = get_dur_mapping(drug_a)
    map_b = get_dur_mapping(drug_b)

    if not map_a or not map_b:
        return {
            "checked": False,
            "is_taboo": False,
            "reason": "DUR 매핑 테이블에 등록되지 않은 약물입니다 (수동 매핑 미작성).",
        }

    if map_a.get("note") == "biologic" or map_b.get("note") == "biologic":
        return {
            "checked": True,
            "is_taboo": False,
            "drug_a_mapped": map_a,
            "drug_b_mapped": map_b,
            "reason": "생물학적제제(바이오의약품)는 DUR 병용금기 데이터베이스(경구약 중심) 대상이 아닙니다.",
        }

    items = query_dur_taboo(map_a["kr_name"], api_key)

    # 응답 항목 중 상대 약물(drug_b)의 영문 성분명이 MIXTURE_INGR_ENG_NAME(병용금기 대상 성분)에
    # 포함되어 있는지 확인. 실제 API 응답 확인 결과 상대 성분 정보는
    # MIXTURE_INGR_ENG_NAME / MIXTURE_INGR_KOR_NAME / MIXTURE_ITEM_NAME 필드에 담겨 있음.
    target_ingr = map_b["ingr_eng"].upper()
    matched_item = None
    for it in items:
        haystack = " ".join(
            [
                str(it.get("MIXTURE_INGR_ENG_NAME", "")),
                str(it.get("MIXTURE_INGR_KOR_NAME", "")),
                str(it.get("MIXTURE_ITEM_NAME", "")),
            ]
        ).upper()
        if target_ingr in haystack:
            matched_item = it
            break

    return {
        "checked": True,
        "is_taboo": matched_item is not None,
        "drug_a_mapped": map_a,
        "drug_b_mapped": map_b,
        "raw_match": matched_item,
        "reason": (
            f"한국 식약처 DUR 병용금기 기준에서 확인됨 — 금기사유: {matched_item.get('PROHBT_CONTENT', '-')}"
            if matched_item
            else "조회 결과, 한국 DUR 병용금기 목록에서 해당 조합은 확인되지 않았습니다."
        ),
    }
