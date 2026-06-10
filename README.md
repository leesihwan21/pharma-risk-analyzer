# 💊 Pharma Risk Analyzer

> **AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System**
> FDA FAERS 데이터 기반 + XGBoost 위험도 예측 + SHAP XAI + RAG 약물 Q&A + PubMed 논문 연동 + ICH E2B(R3) + 21 CFR Part 11

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![FAERS](https://img.shields.io/badge/Data-FDA%20FAERS%202024Q1--2025Q1-red)](https://www.fda.gov/drugs/surveillance/fdas-adverse-event-reporting-system-faers)
[![XAI](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)
[![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20LangChain-blueviolet)](https://github.com/facebookresearch/faiss)
[![CFR](https://img.shields.io/badge/Compliance-21%20CFR%20Part%2011-green)](https://www.fda.gov)

---

## 🌐 Live Demo

**배포 URL**: https://pharma-risk-analyzer-production.up.railway.app

---

## 📌 프로젝트 개요 | Overview

FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 다분기 데이터(약 480,000건)를 기반으로, 약물별 이상반응 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다.

**핵심 특징:**
- SHAP 기반 설명 가능한 AI(XAI)로 예측 근거 시각화
- PubMed 논문 FAISS 벡터DB 임베딩 → RAG 기반 약물 안전성 Q&A 챗봇
- PRR(Evans) + EBGM(FDA MGPS 베이지안) + MedDRA SOC 분류 3종 신호 탐지
- ICH E2B(R3) XML 자동 생성 + 21 CFR Part 11 전자서명
- 식약처 낙알식별 API + OpenFDA 연동 약품 정보 조회

---

## ✨ 주요 기능 | Key Features

### 📊 분석 & 예측
| 기능 | 설명 |
|------|------|
| Dashboard | FAERS 데이터 기반 이상반응 통계 시각화 (6개 차트) |
| AI 위험도 예측 | 약물·이상반응·나이·성별 입력 → XGBoost 고위험/저위험 판정 |
| SHAP XAI | 예측 결과 특성 기여도 시각화 (설명 가능한 AI) |
| 분기별 트렌드 | 2024 Q1~2025 Q1 분기별 이상반응 보고 추이 |

### 💊 약물 검색 & 정보
| 기능 | 설명 |
|------|------|
| Drug Lookup | 약품명·모양/색상으로 식약처+OpenFDA 정보 조회 + AI 안전성 리포트 PDF |
| Interaction Checker | FDA FAERS 기반 약물 병용 시 이상반응 위험 분석 |
| Drug Comparison | 두 약물 통계 나란히 비교 + AI 안전성 리포트 연동 |
| Data Filter | 약물명·성별·나이·결과·국가 조건 필터링 |
| Dosage Calculator | CrCl·소아용량·BSA 항암제·mg/kg 임상약학 용량 계산 |

### 🔬 RAG & AI 문헌 분석
| 기능 | 설명 |
|------|------|
| RAG 약물 Q&A | PubMed 30개 약물 1,689 청크 → FAISS 벡터DB → llama3.2 근거 기반 답변 + 히스토리 DB 저장 |
| AI 안전성 리포트 | FDA FAERS + PubMed 5편 통합 → 6개 섹션 자동 생성 + PDF 다운로드 |
| 논문 검색 | PubMed API 논문 검색 + Claude AI 한국어 요약 |

### 📡 신호 탐지 & 규제 준수
| 기능 | 설명 |
|------|------|
| PRR 신호 탐지 | Evans 기준 (PRR ≥ 2, n ≥ 3) 약물 이상반응 신호 탐지 |
| EBGM 신호 탐지 | FDA MGPS 베이지안 알고리즘 근사 (EB05 ≥ 2 기준) |
| MedDRA SOC 분류 | System Organ Class 기반 부작용 체계 분류 + 파이 차트 |
| AE Manager | CTCAE 자동 등급화·SAE 판정·15일 보고 마감·ICH E2B(R3) XML |
| 21 CFR Part 11 | SHA-256 전자서명·비밀번호 재확인·서명 이력 DB 저장 |
| Audit Trail | 모든 AE 데이터 생성/수정/삭제/내보내기 이력 자동 기록 |

---

## 🔬 RAG 파이프라인 | RAG Pipeline

```
사용자 질문
    ↓
sentence-transformers 임베딩 (all-MiniLM-L6-v2)
    ↓
FAISS 벡터DB 유사도 검색 (Top-3 청크)
    ↓
llama3.2 (Ollama) 컨텍스트 기반 답변 생성
    ↓
근거 논문 청크 + Q&A 히스토리 DB 저장
```

- **벡터DB**: FAISS (30개 약물, 1,689개 청크)
- **임베딩 모델**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: llama3.2 (Ollama 로컬)
- **데이터 소스**: PubMed E-utilities API (무료)

---

## 📡 신호 탐지 알고리즘 비교

| 알고리즘 | 기준 | 특징 |
|---------|------|------|
| PRR | Evans: PRR ≥ 2, n ≥ 3 | 빠르고 직관적, FDA/EMA 표준 |
| EBGM | EB05 ≥ 2, n ≥ 3 | 소표본 보정, FDA MGPS 베이지안 |
| MedDRA SOC | System Organ Class | 신체 기관계별 체계 분류 |

---

## 🔐 규제 준수 기능 | Regulatory Compliance

### ICH E2B(R3) XML
- `<primarysource>` 보고자 정보
- `<sender>` / `<receiver>` MFDS 수신자
- `<reactionmeddraversionpt>26.1` MedDRA 버전
- `<narrativeincludeclinical>` 구조화된 Narrative
- Audit Trail 자동 기록

### 21 CFR Part 11 전자서명
- SHA-256 해시 기반 전자서명
- 비밀번호 재확인 (신원 인증)
- 서명 의미 + 서명 사유 필수 입력
- 서명 이력 DB 영구 보존
- 서명 후 AE 제출 상태 자동 업데이트

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend    : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI      : XGBoost, SHAP (XAI), YOLOv8 (Ultralytics)
RAG        : LangChain, FAISS, sentence-transformers, llama3.2 (Ollama)
Data       : FDA FAERS 2024 Q1~2025 Q1 (~480,000건), 식약처 이상반응 데이터
External   : 식약처 낙알식별 OpenAPI, OpenFDA Drug Label API, PubMed E-utilities API
Viz        : Plotly, NetworkX (Canvas), Chart.js
DB         : SQLite (개발/배포)
Report     : ReportLab (PDF), ICH E2B(R3) XML
Compliance : 21 CFR Part 11 전자서명, Audit Trail, ICH E2B(R3)
Frontend   : Jinja2 Templates, Vanilla JS, 반응형 CSS
Deploy     : Railway
Test       : pytest (28개 유닛테스트)
```

---

## 🗂️ 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py          # User, AEReport, AuditTrail, ElectronicSignature, RagHistory 등
│   └── routes/
│       ├── main.py
│       ├── drug.py        # Drug Lookup + AI 안전성 리포트 + PDF
│       ├── ae.py          # AE Manager + ICH E2B(R3) + 21 CFR Part 11
│       ├── analysis.py    # PRR + EBGM + MedDRA SOC + SHAP
│       ├── auth.py
│       ├── vision.py      # YOLOv8 알약 탐지 + 실시간 웹캠 + 이미지 업로드
│       ├── literature.py
│       └── rag.py         # RAG Q&A (FAISS + LangChain + 히스토리 DB)
├── data/
│   ├── processed/
│   │   └── processed_faers.csv
│   └── download_faers.py
├── ml/
│   ├── model.pkl
│   ├── le_drug.pkl
│   ├── le_reac.pkl
│   └── risk_rates.pkl
├── build_rag.py           # RAG 벡터DB 구축 스크립트
├── config.py
├── run.py
└── README.md
```

---

## 🚀 설치 및 실행 | Installation & Run

```bash
# 1. 저장소 복제
git clone https://github.com/leesihwan21/pharma-risk-analyzer.git
cd pharma-risk-analyzer

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate        # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (app/.env 파일 생성)
SECRET_KEY=your-secret-key
MFDS_API_KEY=your-mfds-api-key
ANTHROPIC_API_KEY=your-api-key

# 5. 다분기 데이터 다운로드
python data/download_faers.py

# 6. 데이터 전처리
python data/preprocess.py

# 7. ML 모델 학습
python ml/train_model.py

# 8. RAG 벡터DB 구축 (선택 - Ollama 필요)
python build_rag.py

# 9. 서버 실행
python run.py
```

브라우저에서 `http://127.0.0.1:5001` 접속

---

## 📊 ML 모델 | ML Model Specifications

- **알고리즘**: XGBoost Classifier
- **학습 특성**: `drugname_enc`, `reaction_enc`, `sex_enc`, `age`, `drug_risk_rate`, `reac_risk_rate`, `combo_risk_rate`
- **예측 목표**: Serious Outcome (입원/사망/장애 → 1, 경미 → 0)
- **성능**: Accuracy 69.4% (480,000건 기준)
- **설명가능성**: SHAP TreeExplainer 기반 특성 기여도 시각화

> XGBoost 선택 이유: 테이블형 데이터에서 SHAP 설명가능성이 뛰어나고, 48만 건 규모에서 학습 속도와 성능이 균형적.

---

## 📂 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 약물 이상반응 자발 보고 데이터
- **식약처 이상반응**: 연도별(2019~2024) 국내 이상 보고 통계
- **식약처 낙알식별 OpenAPI**: 공공데이터포털(data.go.kr)
- **OpenFDA Drug Label API**: FDA 공식 약물 설명서
- **PubMed E-utilities API**: NCBI 논문 검색 및 초록 수집 (무료)

---

## ⚠️ 면책조항 | Disclaimer

본 애플리케이션은 **연구·교육·포트폴리오 목적**으로 제작되었으며, 실제 임상 처방결정을 위해 사용하면 안 됩니다.

This tool is built for **research and portfolio purposes only**.

---

## 👨‍💻 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 예정 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
