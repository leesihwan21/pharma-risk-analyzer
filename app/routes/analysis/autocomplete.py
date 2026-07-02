"""
app/routes/analysis/autocomplete.py
약물명 / 부작용명 자동완성
"""
from flask import jsonify, request
# jsonify : 파이썬 데이터 (리스트, 딕셔너리) 를 JSON 형태로 변환해서 응답하는 함수
# request : 사용자가 보낸 요청 데이터 (URL 파라미터 등) 을 읽기 위한 객체

from . import analysis
# 같은 폴더 (analysis) 안의 __init__.py 에서 이미 만들어 준 블루프린트(analysis)를 가져옴
# 블루프린트 : 라우트들을 그룹으로 묶어서 관리하는 Flask 기능.

from ._common import load_df
# 같은 폴더의 __common.py 에 있는 load_df 함수를 가져옴.
# 아마 CSV/DB에서 데이터를 pandas DataFrame으로 불러오는 공통 함수


@analysis.route("/api/autocomplete")
# analysis 블루프린트에 "/api/autocomplete" 경로 등록하는 함수
# 실제 접속 주소는 앞에 블루프린트 prefix 가 붙음. 
def api_autocomplete():
    q = request.args.get("q", "").upper().strip()
    # URL의 쿼리 파라미터에서 "Q" 값을 꺼내옴. (예: /api/autocomplete?q=asp)
    # 없으면 빈 문자열 ("") 기본값으로 사용
    # .upper() : 대문자로 통일 (대소문자 구분 없이 검색되게)
    # .strip() : 앞뒤 공백 제거 함수

    if len(q) < 2:
        return jsonify([])
    try:
        df = load_df()
        # 약물 데이터 전체를 DataFrame으로 불러옴.
        drugs = df["drugname"].str.upper().dropna().unique()
        # drugname 컬럼 값을 다 대문자로 바꾸고
        # dropna() : 빈 값 제거
        # unique() : 중복 제거해서 고유한 약물명만 남기는 함수
        matched = sorted([d for d in drugs if d.startswith(q)])[:10]
        # 검색어(q) 로 "시작하는" 약물명만 골라서 (리스트 컴프리헨션)
        # sorted() : 알파벳순으로 정렬
        # [:10] : 최대 10개까지만 자름 (너무 많이 보내지 않으려고)
        return jsonify(matched)
        # 매칭된 리스트를 JSON 형태로 반환해서 응답
    except Exception:
        return jsonify([])
    # 에러가 나면 (예 : 데이터 로드 실패) 그냥 빈 리스트 반환
    # 자동완성 기능이라 에러가 나도 화면이 깨지면 안 되니까 조용히 무시함.


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
