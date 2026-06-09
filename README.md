# 💊 Pharma Risk Analyzer

### AI 기반 약물 이상반응 위험도 분석 및 임상 지원 시스템
> AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System

> **AI-powered Drug Adverse Event Risk Analysis System**
> FDA FAERS 데이터셋 + YOLOv8 알약 탐지 + XGBoost 위험도 예측 + SHAP 설명가능 AI + 식약처 약물 조회 + 복용량 계산기

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![FAERS](https://img.shields.io/badge/Data-FDA%20FAERS%202024Q1--2025Q1-red)](https://www.fda.gov/drugs/surveillance/fdas-adverse-event-reporting-system-faers)
[![XAI](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)
[![MFDS](https://img.shields.io/badge/API-%EC%8B%9D%EC%95%BD%EC%B2%98%20%EB%82%B1%EC%95%8C%EC%8B%9D%EB%B3%84-brightgreen)](https://www.mfds.go.kr)

---

## 🔗 Live Demo

**배포 URL**: https://pharma-risk-analyzer-production.up.railway.app

---

## 📦 프로젝트 개요 | Overview

**한국어**
FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 다분기 데이터셋을 기반으로, 약물별 이상반응 발생 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다. SHAP 기반 설명가능 AI(XAI)로 예측 근거를 시각화하며, 식약처 낱알식별 API와 OpenFDA를 활용해 한국/미국 약물 정보를 조회합니다. 임상약학 기반 복용량 계산기(CrCl, 소아용량, BSA)도 포함합니다.

**English**
A web application analyzing drug adverse event patterns and predicting risk levels using XGBoost ML, based on real-world FDA FAERS multi-quarter data (2024 Q1 ~ 2025 Q1). Features SHAP-based explainable AI, quarterly trend analysis, integrated Korean MFDS + OpenFDA drug lookup, drug-drug interaction checker, and clinical pharmacy dosage calculators.

---

## ✨ 주요 기능 | Key Features

| 기능 | 설명 | Feature |
|------|------|---------|
| 📊 대시보드 | FAERS 데이터 기반 이상반응 통계 시각화(6개 차트) | Adverse event statistics dashboard |
| 🔍 약물 검색 | 약물명 자동완성 + 상세 이상반응 분석 | Drug search with autocomplete |
| 🎯 AI 위험도 예측 | 약물·이상반응·나이·성별 입력 → 위험도 분류 (XGBoost) | XGBoost-based risk prediction |
| 🔍 SHAP 설명가능 AI | 예측 근거 특성 기여도 시각화 | SHAP-based XAI feature importance |
| 📈 분기별 트렌드 | 2024 Q1~2025 Q1 분기별 이상반응 보고 추이 | Quarterly adverse event trend |
| 💊 Drug Lookup | 약물명 낱알/성분으로 식약처+OpenFDA 정보 조회 | Korean MFDS + OpenFDA drug lookup |
| ⚡ Interaction Checker | FDA FAERS 기반 두 약물 동시 복용 시 이상반응 패턴 분석 | Drug-drug interaction analysis |
| 🧮 복용량 계산기 | CrCl(신장기능) · 소아용량 · BSA 체표면적 · 기본 mg/kg 계산 | Clinical dosage calculator |
| 💊 알약 이미지 탐지 | YOLOv8으로 알약 이미지 자동 판별 및 위험도 분석 | YOLOv8 pill detection |
| 📸 실시간 웹캠 탐지 | 웹캠으로 실시간 알약 탐지 | Real-time webcam pill detection |
| 🆚 약물 비교 | 두 약물의 이상반응 패턴 비교 | Side-by-side drug comparison |
| 🕸 이상반응 네트워크 | 약물-이상반응 연관 네트워크 그래프 | Drug-reaction network graph |
| 📄 PDF 보고서 | 약물 분석 결과 PDF 자동 생성 | Automated PDF report generation |
| 🇰🇷 한국 데이터 | 식약처 국내 이상반응 연도별 트렌드 분석 | Korean MFDS adverse event trends |
| 👤 회원 기능 | 로그인/회원가입/즐겨찾기/기록 | User auth, favorites, history |
| 📋 AE Manager | CTCAE 자동분류·SAE 판정·15일 보고 체크·PDF/E2B 내보내기 | AE management with CTCAE auto-grading |
| 📡 PRR 신호 탐지 | FDA/EMA Evans 기준 PRR 계산 및 시각화 | PRR-based signal detection |
| 📎 ICH E2B XML | 국제기준 준수 ICH E2B(R3) 형식 XML 자동 생성 | ICH E2B(R3) XML export |
| 🌐 한국어/영어 | 전체 페이지 한국어↔영어 실시간 전환 | Korean/English i18n |

---

## 🧮 복용량 계산기 | Dosage Calculator

임상약학에서 실제 사용하는 공식을 기반으로 용량 계산 제공합니다.

| 구분 | 공식 | 용도 |
|---|---|---|
| 신장기능 (CrCl) | Cockcroft-Gault | 신장대사 약물 용량 조정 및 투석환자 용량 결정 참고 |
| 소아 용량 | Clark / Young / BSA | 성인 용량 기준 소아 용량 환산 (3가지 공식 비교) |
| BSA 체표면적 | Mosteller / DuBois | 항암화학 계산 및 mg/m² 기준 용량 환산 |
| 기본 용량 | mg/kg | 체중 기반 용량 계산 + 최대 용량 적용 |

> ⚠️ **중요**: 본 계산기는 **교육·연구·포트폴리오 목적**입니다.
> 실제 약물 처방 결정은 반드시 **의사 또는 약사에게 직접 문의**하세요.

---

## 🔧 기술 스택 | Tech Stack

```
Backend   : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI     : XGBoost, SHAP (XAI), YOLOv8 (Ultralytics)
Data      : FDA FAERS 2024 Q1~2025 Q1 (다분기 480,000행, 한국 식약처 이상반응 데이터)
External  : 식약처 낱알식별 OpenAPI, OpenFDA Drug Label API
Viz       : Plotly, NetworkX (Canvas), Chart.js
DB        : SQLite (개발/배포), PyMySQL 지원
Report    : ReportLab (PDF), ICH E2B(R3) XML
Frontend  : Jinja2 Templates, Vanilla JS, 커스텀 CSS
Deploy    : Railway
i18n      : 한국어/영어(lang.js, localStorage)
Test      : pytest (28개 테스트)
```

---

## 📁 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes/
│   │   ├── main.py
│   │   ├── drug.py
│   │   ├── ae.py
│   │   ├── analysis.py
│   │   ├── auth.py
│   │   └── vision.py
│   └── templates/
├── data/
│   ├── raw/
│   │   └── korea_adr.csv
│   ├── processed/
│   │   └── processed_faers.csv
│   ├── download_faers.py
│   └── preprocess.py
├── ml/
│   ├── model.pkl
│   ├── le_drug.pkl
│   ├── le_reac.pkl
│   └── risk_rates.pkl
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

# 8. 서버 실행
python run.py
```

브라우저에서 `http://127.0.0.1:5001` 접속

---

## 🤖 머신러닝 모델 | ML Model Specifications

- **알고리즘**: XGBoost Classifier
- **학습 특성**: `drugname_enc`, `reaction_enc`, `sex_enc`, `age`, `drug_risk_rate`, `reac_risk_rate`, `combo_risk_rate`
- **예측 목표**: Serious Outcome (사망/입원/장애 등 → 1, 경증 → 0)
- **검증 성능**: Accuracy 69.4% (480,000행 데이터셋 기준)
- **설명성**: SHAP TreeExplainer 기반 특성 기여도 시각화

---

## 📊 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 웹사이트 실제 이상반응 자발 보고 데이터
- **한국 식약처 이상반응**: 연도별(2019~2024) 상위 증상 보고 통계
- **식약처 낱알식별 OpenAPI**: 공공데이터포털(data.go.kr)
- **OpenFDA Drug Label API**: FDA 공식 약물 성분 정보

---

## ⚠️ 면책조항 | Disclaimer

본 도구는 **교육·포트폴리오 목적**으로 제작되었으며, 실제 임상에서 의사결정용으로 사용하면 안 됩니다.
복용량 계산기를 포함한 모든 기능은 교육 목적이며, **실제 약물 처방은 반드시 의사 또는 약사에게 문의**하세요.

This tool is built for **research and portfolio purposes only** and should not be used for actual clinical decision-making.

---

## 👨‍💻 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
