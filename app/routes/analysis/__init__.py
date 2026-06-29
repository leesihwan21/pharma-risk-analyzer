"""
app/routes/analysis/ 패키지
기능별로 모듈을 분리하고 Blueprint를 통합 등록합니다.
"""
from flask import Blueprint

analysis = Blueprint('analysis', __name__)

# 각 서브모듈 임포트 (Blueprint에 라우트 등록)
from . import prr        # PRR, EBGM, emerging signals, favorites alerts
from . import shap_xai   # SHAP, LIME
from . import interaction # drug interaction, polypharmacy
from . import dosage     # CrCl, pediatric, BSA
from . import trend      # trend, Prophet forecast, quarterly PRR
from . import drug_lookup # drug lookup, drug shape, drug vision, SOC, autocomplete

from .prr import compute_prr_summary, compute_emerging_signals

from .shap_xai import compute_shap

