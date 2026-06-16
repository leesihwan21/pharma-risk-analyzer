# 💊 Pharma Risk Analyzer

[![Tests](https://github.com/leesihwan21/pharma-risk-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/leesihwan21/pharma-risk-analyzer/actions/workflows/tests.yml)

> **AI-powered Drug Adverse Event Risk Analysis & Clinical Decision Support System**
> FDA FAERS 데이터 기반 + XGBoost 위험도 예측 + SHAP/LIME XAI + RAG 약물 Q&A + PubMed 논문 연동 + ICH E2B(R3) + 21 CFR Part 11 + MLOps 파이프라인

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20Optuna-orange)](https://xgboost.readthedocs.io)
[![FAERS](https://img.shields.io/badge/Data-FDA%20FAERS%202024Q1--2025Q1-red)](https://www.fda.gov/drugs/surveillance/fdas-adverse-event-reporting-system-faers)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-purple)](https://shap.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow%20%2B%20Prophet-blueviolet)](https://mlflow.org)
[![CFR](https://img.shields.io/badge/Compliance-21%20CFR%20Part%2011-green)](https://www.fda.gov)

---

## 🌐 Live Demo

**배포 URL**: [http://3.36.178.61:5001](http://3.36.178.61:5001)

---

## 📌 프로젝트 개요 | Overview

FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 다분기 데이터(약 480,000건)를 기반으로, 약물별 이상반응 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다.

**핵심 특징:**
- Optuna 하이퍼파라미터 자동 튜닝 + SHAP/LIME 설명 가능한 AI(XAI)
- MLflow 실험 추적 + Prophet 시계열 예측 + 분기별 자동 재학습 파이프라인
- K-Means 약물 클러스터링 + Co-medication 연관 분석 추천 시스템
- PubMed 논문 FAISS 벡터DB 임베딩 → RAG 기반 약물 안전성 Q&A 챗봇
- PRR(Evans) + EBGM(FDA MGPS 베이지안) + MedDRA SOC 분류 3종 신호 탐지
- ICH E2B(R3) XML 자동 생성 + 21 CFR Part 11 전자서명

---

## ✨ 주요 기능 | Key Features

### 📊 분석 & 예측
| 기능 | 설명 |
|------|------|
| Dashboard | FAERS 데이터 기반 이상반응 통계 시각화 (6개 차트) |
| AI 위험도 예측 | 약물·이상반응·나이·성별 입력 → XGBoost 고위험/저위험 판정 |
| SHAP / LIME XAI | SHAP(Global) + LIME(Local) 예측 근거 비교 시각화 |
| 분기별 트렌드 + 예측 | 2024 Q1~2025 Q1 분기별 추이 + Prophet 향후 2개 분기 예측 |

### 🤖 MLOps 파이프라인
| 기능 | 설명 |
|------|------|
| Optuna 튜닝 | 9개 하이퍼파라미터 자동 탐색 (50 trials, CV 3-fold) |
| MLflow 실험 추적 | run별 파라미터/메트릭 자동 기록·비교 |
| ML Dashboard | MLflow 실험 결과 웹 대시보드 (`/ml_dashboard`) |
| 자동 재학습 | 새 FAERS 분기 데이터 자동 다운로드 → 전처리 → 재학습 → MLflow 기록 |

### 💡 추천 시스템
| 기능 | 설명 |
|------|------|
| Drug Clustering | K-Means로 유사 부작용 프로파일 약물 추천 (8개 클러스터) |
| Co-medication 분석 | 함께 복용된 약물 Top 10 + 중증 부작용 비율 분석 |

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
| RAG 약물 Q&A | PubMed 30개 약물 1,689 청크 → FAISS 벡터DB → llama3.2 근거 기반 답변 |
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

## 🧪 ML/MLOps 구성 | ML Pipeline

```
[FDA FAERS 분기 데이터]
        ↓
[전처리 + 피처 엔지니어링]
  drug_risk_rate / reac_risk_rate / combo_risk_rate
        ↓
[Optuna 하이퍼파라미터 튜닝]
  50 trials × CV 3-fold → 최적 파라미터 탐색
        ↓
[XGBoost 학습]
  Accuracy 69.3% | F1(위험) 0.622 | Recall 0.707
        ↓
[MLflow 실험 기록]
  파라미터 / 메트릭 / 모델 아티팩트 자동 저장
        ↓
[SHAP + LIME XAI]
  Global(SHAP) + Local(LIME) 예측 설명
        ↓
[자동 재학습 파이프라인]
  분기마다 새 데이터 → 재학습 → MLflow 비교
```

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend    : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI      : XGBoost, Optuna (하이퍼파라미터 튜닝), SHAP, LIME
MLOps      : MLflow (실험 추적), Prophet (시계열 예측), scikit-learn
추천 시스템 : K-Means 클러스터링, Co-medication 연관 분석
RAG        : LangChain, FAISS, sentence-transformers, llama3.2 (Ollama)
Data       : FDA FAERS 2024 Q1~2025 Q1 (~480,000건)
External   : 식약처 낙알식별 OpenAPI, OpenFDA, PubMed E-utilities API
Viz        : Plotly, Chart.js, NetworkX
DB         : SQLite (mlflow.db 포함)
Report     : ReportLab (PDF), ICH E2B(R3) XML
Compliance : 21 CFR Part 11 전자서명, Audit Trail, ICH E2B(R3)
Frontend   : Jinja2 Templates, Vanilla JS, 반응형 CSS, PWA
Deploy     : Railway
Test       : pytest (92개 유닛테스트)
```

---

## 🗂️ 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   └── routes/
│       ├── analysis.py    # PRR + EBGM + SHAP + LIME + Prophet
│       ├── recommend.py   # K-Means 클러스터링 + Co-medication 분석
│       ├── ml_dashboard.py # MLflow 실험 대시보드
│       ├── drug.py
│       ├── ae.py
│       ├── auth.py
│       ├── vision.py
│       ├── literature.py
│       └── rag.py
├── ml/
│   ├── train_model_optuna.py  # Optuna + MLflow 학습
│   ├── retrain_pipeline.py    # 분기별 자동 재학습
│   ├── model.pkl
│   ├── best_params.json
│   └── pipeline_log.json
├── data/
│   └── processed/processed_faers.csv
├── mlflow.db              # MLflow 실험 DB
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
venv\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (.env)
SECRET_KEY=your-secret-key
MFDS_API_KEY=your-mfds-api-key
ANTHROPIC_API_KEY=your-api-key

# 5. ML 모델 학습 (Optuna + MLflow)
python ml/train_model_optuna.py

# 6. 서버 실행
python run.py
# → http://127.0.0.1:5001

# 7. MLflow UI (별도 터미널)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
# → http://127.0.0.1:5002

# 8. 분기별 자동 재학습
python ml/retrain_pipeline.py --quarter 2025q2
```

---

## 📊 ML 모델 성능 | Model Performance

| 지표 | 값 | 설명 |
|------|-----|------|
| Accuracy | 69.3% | 전체 정확도 |
| F1 (위험) | 0.622 | 위험 클래스 F1 |
| Recall (위험) | 0.707 | 위험 케이스 탐지율 |
| Precision (위험) | 0.554 | 위험 예측 정밀도 |
| CV F1 (Optuna) | 0.622 | 50 trials 교차검증 |

> **Recall 우선 최적화**: 의약품 도메인에서 위험 케이스를 놓치지 않는 것이 중요하므로 F1(위험) 최대화로 튜닝.

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

---

## 👨‍💻 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 예정 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
