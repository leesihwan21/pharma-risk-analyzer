"""
app/routes/ae/ 패키지
AE(이상사례) 관리 기능을 모듈별로 분리하고 Blueprint를 통합 등록.
구조:
    _common.py      - 공통 함수/상수 (감사로그, CTCAE 분류, PDF 테이블, 동의문구)
    core.py         - AE CRUD (등록/조회/수정/삭제/제출)
    stats.py        - 통계 대시보드
    export.py       - PDF, E2B XML 생성
    signature.py    - 전자서명, 감사로그 조회
"""

# ruff : noqa: F401

from flask import Blueprint

ae = Blueprint("ae", __name__)

from . import core
from . import stats
from . import export
from . import signature