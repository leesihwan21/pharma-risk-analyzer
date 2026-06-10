# 💊 Pharma Risk Analyzer

> **AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System**
> FDA FAERS 데이터 기반 + XGBoost 위험도 예측 + SHAP XAI + RAG 약물 Q&A + PubMed 논문 연동

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![FAERS](https://img.shields.io/badge/Data-FDA%20FAERS%202024Q1--2025Q1-red)](https://www.fda.gov/drugs/surveillance/fdas-adverse-event-reporting-system-faers)
[![XAI](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)
[![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20LangChain-blueviolet)](https://github.com/facebookresearch/faiss)

---

## 🌐 Live Demo

**배포 URL**: https://pharma-risk-analyzer-production.up.railway.app

---

## 📌 프로젝트 개요 | Overview

**한국어**
FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 다분기 데이터(약 480,000건)를 기반으로, 약물별 이상반응 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다. SHAP 기반 설명 가능한 AI(XAI)로 예측 근거를 시각화하며, 식약처 낙알식별 API와 OpenFDA를 연동한 국내/해외 약품 정보 조회, 임상약학 기반 용량 계산기를 포함합니다. 또한 PubMed 논문을 FAISS 벡터DB에 임베딩한 RAG 파이프라인으로 약물 안전성 Q&A 챗봇을 구현하였습니다.

**English**
A web application analyzing drug adverse event patterns and predicting risk levels using XGBoost ML, based on real-world FDA FAERS multi-quarter data (2024 Q1 ~ 2025 Q1, ~480K records). Features SHAP-based explainable AI, RAG-based drug Q&A chatbot (FAISS + LangChain + llama3.2), PubMed-integrated AI safety reports, Korean MFDS + OpenFDA drug lookup, drug-drug interaction checker, and clinical pharmacy dosage calculators.

---

## ✨ 주요 기능 | Key Features

| 기능 | 설명 | Feature |
|------|------|---------|
| 📊 대시보드 | FAERS 데이터 기반 이상반응 통계 시각화 (6개 차트) | Adverse event statistics dashboard |
| 🔍 약물 검색 | 약품명 자동완성 + 상세 이상반응 분석 | Drug search with autocomplete |
| 🤖 AI 위험도 예측 | 약물·이상반응·나이·성별 입력 → 위험도 예측 (XGBoost) | XGBoost-based risk prediction |
| 🔍 SHAP 설명가능 AI | 예측 결과 특성 기여도 시각화 | SHAP-based XAI feature importance |
| 📈 분기별 트렌드 | 2024 Q1~2025 Q1 분기별 이상반응 보고 추이 | Quarterly adverse event trend |
| 💊 Drug Lookup | 약품명·모양/색상으로 식약처/OpenFDA 정보 조회 | Korean MFDS + OpenFDA drug lookup |
| 🧬 AI 안전성 리포트 | PubMed 논문 5편 + FDA FAERS 통합 → llama3.2 안전성 리포트 자동 생성 | AI safety report (PubMed + FAERS) |
| 🔬 RAG 약물 Q&A | PubMed 논문 FAISS 벡터DB 임베딩 → LangChain RAG 기반 약물 안전성 챗봇 | RAG-based drug safety Q&A chatbot |
| ⚗️ Interaction Checker | FDA FAERS 기반 약물 병용 시 이상반응 위험 분석 | Drug-drug interaction analysis |
| 💉 용량 계산기 | CrCl(신기능) · 소아용량 · BSA 항암제 · 표준 mg/kg 계산 | Clinical dosage calculator |
| 💊 알약 이미지 탐지 | YOLOv8으로 알약 이미지 자동 감지 및 위험도 분석 | YOLOv8 pill detection |
| 📋 AE Manager | CTCAE 자동등급화·SAE 판정·15일 보고 체크·PDF/E2B XML 내보내기 | AE management with CTCAE auto-grading |
| 📉 PRR 신호 탐지 | FDA/EMA Evans 기준 PRR 계산 및 시각화 | PRR-based signal detection |
| 🗂️ ICH E2B XML | 국제규정 준수 ICH E2B(R3) 형식 XML 자동 생성 | ICH E2B(R3) XML export |
| 🌐 한국어/영어 | 전체 페이지 한국어/영어 실시간 전환 | Korean/English i18n |
| 👤 사용자 기능 | 로그인·즐겨찾기·기록/이력 관리 | User auth, favorites, history |

---

## 🔬 RAG 파이프라인 | RAG Pipeline

PubMed에서 수집한 약물 관련 논문을 FAISS 벡터DB에 임베딩하여 질문 기반 검색 후 LLM 답변을 생성합니다.

```
사용자 질문
    ↓
sentence-transformers 임베딩 (all-MiniLM-L6-v2)
    ↓
FAISS 벡터DB 유사도 검색 (Top-3 청크)
    ↓
llama3.2 (Ollama) 컨텍스트 기반 답변 생성
    ↓
근거 논문 청크 함께 반환
```

- **벡터DB**: FAISS (247개 청크, 5개 약물)
- **임베딩 모델**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: llama3.2 (Ollama 로컬)
- **데이터 소스**: PubMed E-utilities API (무료)

---

## 🧬 AI 안전성 리포트 | AI Safety Report

Drug Lookup 페이지의 "🧬 AI 안전성 리포트" 탭에서 약물명 입력 시 자동 생성됩니다.

```
약물명 입력
    ↓
FDA FAERS 통계 추출 (총 보고건수, 주요부작용 TOP5, 사망/입원 건수)
    ↓
PubMed API 관련 논문 5편 자동 검색
    ↓
llama3.2로 6개 섹션 안전성 리포트 생성
(약물개요 / 부작용분석 / 논문근거 / 고위험군 / 임상권고 / 결론)
```

---

## 💉 용량 계산기 | Dosage Calculator

임상약학에서 실제 사용하는 공식 기반으로 용량 계산을 지원합니다.

| 항목 | 공식 | 용도 |
|---|---|---|
| 신기능 (CrCl) | Cockcroft-Gault | 신기능별 약물 용량 조정 및 신독성 약물 주의 |
| 소아 용량 | Clark / Young / BSA | 소아 용량 기준 소아 용량 산출 (3가지 공식 비교) |
| BSA 항암제 | Mosteller / DuBois | 체표면적 계산 및 mg/m² 기준 용량 산출 |
| 표준 용량 | mg/kg | 체중 기반 용량 계산 + 최대 용량 적용 |

> ⚠️ **중요**: 본 계산기는 **연구·교육·포트폴리오 목적**입니다.
> 실제 약물 처방·조제 결정은 반드시 **의사 또는 약사에게 직접 의뢰**하십시오.

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend    : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI      : XGBoost, SHAP (XAI), YOLOv8 (Ultralytics)
RAG        : LangChain, FAISS, sentence-transformers, llama3.2 (Ollama)
Data       : FDA FAERS 2024 Q1~2025 Q1 (다분기 480,000건), 한국 식약처 이상반응 데이터
External   : 식약처 낙알식별 OpenAPI, OpenFDA Drug Label API, PubMed E-utilities API
Viz        : Plotly, NetworkX (Canvas), Chart.js
DB         : SQLite (개발/배포), PyMySQL 지원
Report     : ReportLab (PDF), ICH E2B(R3) XML
Frontend   : Jinja2 Templates, Vanilla JS, 반응형 CSS
Deploy     : Railway
i18n       : 한국어/영어 (lang.js, localStorage)
Test       : pytest (28개 유닛테스트)
```

---

## 🗂️ 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   └── routes/
│       ├── main.py
│       ├── drug.py       # Drug Lookup + AI 안전성 리포트
│       ├── ae.py
│       ├── analysis.py
│       ├── auth.py
│       ├── vision.py
│       ├── literature.py
│       └── rag.py        # RAG Q&A (FAISS + LangChain)
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── processed_faers.csv
│   ├── download_faers.py
│   └── preprocess.py
├── ml/
│   ├── model.pkl
│   ├── le_drug.pkl
│   ├── le_reac.pkl
│   └── risk_rates.pkl
├── rag_db/               # FAISS 벡터DB (로컬 전용)
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

# 8. RAG 벡터DB 구축 (선택)
python build_rag.py

# 9. 서버 실행
python run.py
```

브라우저에서 `http://127.0.0.1:5001` 접속

---

## 📊 머신러닝 모델 | ML Model Specifications

- **알고리즘**: XGBoost Classifier
- **학습 특성**: `drugname_enc`, `reaction_enc`, `sex_enc`, `age`, `drug_risk_rate`, `reac_risk_rate`, `combo_risk_rate`
- **예측 목표**: Serious Outcome (입원/사망/장애 등 → 1, 경미한 증상 → 0)
- **성능 지표**: Accuracy 69.4% (480,000건 데이터 기준)
- **설명가능성**: SHAP TreeExplainer 기반 특성 기여도 시각화

> XGBoost를 선택한 이유: 테이블형 데이터에서 딥러닝 대비 설명가능성(SHAP)이 뛰어나고, 데이터 규모(48만 건)에서 학습 속도와 성능이 균형적임.

---

## 📂 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 약물 이상반응 자발 보고 데이터
- **한국 식약처 이상반응**: 연도별(2019~2024) 국내 이상 보고 통계
- **식약처 낙알식별 OpenAPI**: 공공데이터포털(data.go.kr)
- **OpenFDA Drug Label API**: FDA 공식 약물 설명서 정보
- **PubMed E-utilities API**: NCBI 논문 검색 및 초록 수집 (무료)

---

## ⚠️ 면책조항 | Disclaimer

본 애플리케이션은 **연구·교육·포트폴리오 목적**으로 제작되었으며, 실제 임상에서 처방결정을 위해 사용하면 안 됩니다.
용량 계산기를 포함한 모든 기능은 연구 목적이며, **실제 약물 처방 및 조제 결정은 의사 또는 약사에게 직접 의뢰**하십시오.

This tool is built for **research and portfolio purposes only** and should not be used for actual clinical decision-making.

---

## 👨‍💻 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 예정 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
