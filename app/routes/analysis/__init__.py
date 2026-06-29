"""
app/routes/analysis/ 패키지
기능별로 모듈을 분리하고 Blueprint를 통합 등록합니다.

구조:
    prr.py          - PRR 신호 탐지, EBGM, Favorites Alerts
    signals.py      - Emerging Signals, 분기별 PRR 급변 탐지
    shap_xai.py     - SHAP, LIME XAI 설명
    trend.py        - 단순 분기별 트렌드 조회
    forecast.py     - Prophet 시계열 예측
    interaction.py  - 약물 상호작용 (2약물, Polypharmacy)
    dosage.py       - 신장기능(CrCl), 소아용량, BSA
    drug_lookup.py  - Drug Lookup, Drug Shape, Drug Vision, SOC
    autocomplete.py - 약물명/부작용명 자동완성
"""
from flask import Blueprint

analysis = Blueprint("analysis", __name__)

from . import prr          # PRR, EBGM, Favorites Alerts
from . import signals      # Emerging Signals, 분기별 PRR 급변
from . import shap_xai     # SHAP, LIME
from . import trend        # 단순 트렌드 조회
from . import forecast     # Prophet 예측
from . import interaction  # 약물 상호작용, Polypharmacy
from . import dosage       # CrCl, 소아용량, BSA
from . import drug_lookup  # Drug Lookup, Vision, SOC
from . import autocomplete # 자동완성

# 테스트 및 외부 모듈에서 직접 import 가능하도록 re-export
from .prr import compute_prr_summary
from .signals import compute_emerging_signals
from .shap_xai import compute_shap
