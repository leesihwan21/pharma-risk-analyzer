# 💊 Pharma Risk Analyzer

### AI 기반 약물 부작용 위험도 분석 및 임상 지원 시스템 > AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System

> **AI-powered Drug Adverse Event Risk Analysis System**  
> FDA FAERS 실데이터 + YOLOv8 알약 탐지 + XGBoost 위험도 예측 + SHAP 설명가능 AI + 식약처 약물 조회 + 복용량 계산기

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![FAERS](https://img.shields.io/badge/Data-FDA%20FAERS%202024Q1--2025Q1-red)](https://www.fda.gov/drugs/surveillance/fdas-adverse-event-reporting-system-faers)
[![XAI](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)
[![MFDS](https://img.shields.io/badge/API-%EC%8B%9D%EC%95%BD%EC%B2%98%20%EB%82%B1%EC%95%8C%EC%8B%9D%EB%B3%84-brightgreen)](https://www.mfds.go.kr)

---

## 🌐 Live Demo

**배포 URL**: https://pharma-risk-analyzer.onrender.com
> ⚠️ 무료 플랜 사용 중으로 첫 접속 시 50초 정도 로딩될 수 있습니다.  
> ML/YOLO 기능(AI 예측, 알약 탐지, 웹캠)은 배포 버전에서 제외됩니다.

---

## 📌 프로젝트 개요 | Overview

**한국어**  
FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 멀티쿼터 실데이터를 기반으로, 약물별 부작용 발생 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다. SHAP 기반 설명가능 AI(XAI)로 예측 근거를 시각화하며, 식약처 낱알식별 API와 OpenFDA를 통해 한국/미국 약물 정보를 통합 제공합니다. 임상약학 공식 기반 복용량 계산기(CrCl, 소아용량, BSA)도 포함합니다.

**English**  
A web application analyzing drug adverse event patterns and predicting risk levels using XGBoost ML, based on real-world FDA FAERS multi-quarter data (2024 Q1 ~ 2025 Q1). Features SHAP-based explainable AI, quarterly trend analysis, integrated Korean MFDS + OpenFDA drug lookup, drug-drug interaction checker, and clinical pharmacy dosage calculators.

---

## 🚀 주요 기능 | Key Features

| 기능 | 설명 | Feature |
|------|------|---------|
| 📊 대시보드 | FAERS 데이터 기반 부작용 통계 시각화 (6개 차트) | Adverse event statistics dashboard |
| 🔍 약물 검색 | 약물명 자동완성 + 상세 부작용 분석 | Drug search with autocomplete |
| 🤖 AI 위험도 예측 | 약물·부작용·나이·성별 입력 → 위험도 분류 (XGBoost) | XGBoost-based risk prediction |
| 🔍 SHAP 설명가능 AI | 예측 근거 피처 기여도 시각화 | SHAP-based XAI feature importance |
| 📈 쿼터별 트렌드 | 2024 Q1~2025 Q1 분기별 부작용 보고 추이 | Quarterly adverse event trend |
| 💊 Drug Lookup | 약물명/모양/색상으로 식약처+OpenFDA 통합 조회 | Korean MFDS + OpenFDA drug lookup |
| ⚡ Interaction Checker | FDA FAERS 기반 두 약물 동시 복용 시 부작용 패턴 분석 | Drug-drug interaction analysis |
| 💉 복용량 계산기 | CrCl(신장기능) · 소아용량 · BSA 항암제 · 기본 mg/kg 계산 | Clinical dosage calculator |
| 📸 알약 이미지 탐지 | YOLOv8으로 알약 종류 자동 인식 후 위험도 분석 | YOLOv8 pill detection |
| 📹 실시간 웹캠 탐지 | 웹캠으로 실시간 알약 탐지 | Real-time webcam pill detection |
| ⚖️ 약물 비교 | 두 약물의 부작용 패턴 나란히 비교 | Side-by-side drug comparison |
| 🔗 부작용 네트워크 | 약물-부작용 관계 네트워크 그래프 | Drug-reaction network graph |
| 📄 PDF 리포트 | 약물 분석 결과 PDF 자동 생성 | Automated PDF report generation |
| 🇰🇷 한국 데이터 | 식약처 이상사례 연도별 트렌드 분석 | Korean MFDS adverse event trends |
| 🔐 회원 기능 | 로그인·즐겨찾기·검색기록 관리 | User auth, favorites, history |
| 🧪 AE Manager | CTCAE 자동분류·SAE 판정·15일 보고 타임라인·PDF/E2B 출력 | AE management with CTCAE auto-grading |
| 📡 PRR 신호 탐지 | FDA/EMA Evans 기준 PRR 계산 및 시각화 | PRR-based signal detection |
| 🗂 ICH E2B XML | 규제기관 제출용 ICH E2B(R3) 형식 XML 자동 생성 | ICH E2B(R3) XML export |
| 🌐 한/영 다국어 | 전체 페이지 한국어·영어 실시간 전환 | Korean/English i18n |

---

## 💉 복용량 계산기 | Dosage Calculator

임상약학에서 실제 사용하는 공식을 기반으로 한 계산 도구입니다.

| 탭 | 공식 | 용도 |
|---|---|---|
| 신장기능 (CrCl) | Cockcroft-Gault | 크레아티닌 청소율 계산 → 신기능 단계 판정 → 용량 조절 가이드 |
| 소아 용량 | Clark / Young / BSA | 성인 용량 기준 소아 용량 산출 (3가지 공식 비교) |
| BSA 항암제 | Mosteller / DuBois | 체표면적 계산 → mg/m² 기준 항암제 총 용량 산출 |
| 기본 용량 | mg/kg | 체중 기반 용량 계산 + 최대 용량 적용 |

> ⚠️ **중요**: 본 계산기는 **교육·연구·포트폴리오 목적 전용**입니다.  
> 실제 약물 투약 결정은 반드시 **의사 또는 약사와 직접 상담**하세요.  
> 실제 임상에서는 간기능, 병용약물, 알레르기, 병력 등 다양한 요소를 종합적으로 고려해야 합니다.

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend   : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI     : XGBoost, SHAP (XAI), YOLOv8 (Ultralytics)
Data      : FDA FAERS 2024 Q1~2025 Q1 (멀티쿼터 480,000행), 한국 식약처 이상사례 데이터
External  : 식약처 낱알식별 OpenAPI, OpenFDA Drug Label API
Viz       : Plotly, NetworkX (Canvas), Chart.js
DB        : SQLite (개발/배포), PyMySQL 지원
Report    : ReportLab (PDF), ICH E2B(R3) XML
Frontend  : Jinja2 Templates, Vanilla JS, 반응형 CSS
Deploy    : Render.com (경량 버전 배포)
i18n      : 한/영 다국어 (lang.js, localStorage)
Test      : pytest (28개 테스트)
```

---

## 📁 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   └── templates/            # HTML 템플릿 (15개 페이지)
│       ├── index.html
│       ├── dashboard.html
│       ├── drug_detail.html
│       ├── compare.html
│       ├── filter.html
│       ├── korea.html
│       ├── webcam.html
│       ├── ae_manager.html
│       ├── prr.html
│       ├── trend.html        # ✅ 쿼터별 트렌드
│       ├── shap.html         # ✅ SHAP XAI 분석
│       ├── drug_lookup.html  # ✅ 약물 정보 통합 조회
│       ├── interaction.html  # ✅ 약물 상호작용 체커
│       ├── dosage.html       # ✅ 복용량 계산기
│       ├── login.html
│       └── register.html
├── data/
│   ├── raw/
│   │   ├── faers_ascii_2024q1/
│   │   ├── faers_ascii_2024q2/
│   │   ├── faers_ascii_2024q3/
│   │   ├── faers_ascii_2025q1/
│   │   └── korea_adr.csv
│   ├── processed/
│   │   └── processed_faers.csv   # 480,000행 멀티쿼터 통합
│   ├── download_faers.py         # 멀티쿼터 자동 다운로드
│   └── preprocess.py             # 메모리 효율적 전처리
├── ml/
│   ├── train_model.py
│   ├── model.pkl
│   └── ...
├── notebooks/
│   └── faers_eda.ipynb           # ✅ 멀티쿼터 EDA 분석
├── config.py
├── run.py
└── README.md
```

---

## ⚙️ 설치 및 실행 | Installation & Run

```bash
# 1. 저장소 클론
git clone https://github.com/leesihwan21/pharma-risk-analyzer.git
cd pharma-risk-analyzer

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate        # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (app/.env 파일 생성)
SECRET_KEY=your-secret-key
MFDS_API_KEY=your-mfds-api-key      # 식약처 공공데이터포털 발급
ANTHROPIC_API_KEY=your-api-key      # 선택사항

# 5. 멀티쿼터 데이터 다운로드
python data/download_faers.py

# 6. 데이터 전처리
python data/preprocess.py

# 7. ML 모델 학습
python ml/train_model.py

# 8. 앱 실행
python run.py
```

브라우저에서 `http://127.0.0.1:5001` 접속

---

## 🧠 머신러닝 파이프라인 | ML Model Specifications

- **알고리즘**: XGBoost Classifier
- **학습 피처**: `drugname_enc`, `reaction_enc`, `sex_enc`, `age`, `drug_risk_rate`, `reac_risk_rate`, `combo_risk_rate`
- **예측 타겟**: Serious Outcome (입원/사망/생명위협 → 1, 기타 → 0)
- **검증 성능**: Accuracy 69.4% (480,000행 실데이터 기준)
- **설명력**: SHAP TreeExplainer 기반 피처 기여도 시각화

### 💡 왜 XGBoost를 선택했나?

FDA FAERS 데이터는 약물명·부작용명·나이·성별 등 **7개의 정형 피처(tabular data)** 로 구성됩니다. 정형 데이터에서는 딥러닝보다 XGBoost/LightGBM 계열이 일반적으로 더 높은 성능을 보이며, 실제 Kaggle 정형 데이터 경진대회에서도 압도적으로 많이 사용됩니다.

딥러닝 대신 XGBoost를 선택한 핵심 이유는 두 가지입니다:

1. **설명가능성 (XAI)**: SHAP TreeExplainer와의 호환성이 뛰어나 예측 근거(피처 기여도)를 직관적으로 시각화할 수 있습니다. "왜 이 약물이 위험한가"를 설명할 수 있는 것이 이 프로젝트의 핵심 강점입니다.

---

## 📊 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 사이트 — 실제 이상사례 자발적 보고 데이터
- **한국 식약처 이상사례**: 연도별(2019~2024) 증상 보고 통계
- **식약처 낱알식별 OpenAPI**: 공공데이터포털(data.go.kr)
- **OpenFDA Drug Label API**: FDA 공식 약물 라벨 정보

---

## ⚠️ 면책조항 | Disclaimer

본 도구는 **연구·포트폴리오 목적**으로 제작되었으며, 실제 임상적 의사결정에 사용해서는 안 됩니다.  
복용량 계산기를 포함한 모든 기능은 교육 목적이며, **실제 약물 투약은 반드시 의사 또는 약사와 상담**하세요.

This tool is built for **research and portfolio purposes only** and should not be used for actual clinical decision-making.  
All features including the dosage calculator are for educational purposes. **Always consult a doctor or pharmacist for actual medication decisions.**

---

## 👤 개발자 | Developer

**이시환 (Sihwan Lee)**  
임상약학 석사 (아주대학교) | AI 개발 교육 (국비, MBC아카데미 수원)  
GitHub: [@leesihwan21](https://github.com/leesihwan21)
