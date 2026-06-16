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
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED)](https://www.docker.com)
[![AWS](https://img.shields.io/badge/Deploy-AWS%20EC2-FF9900)](https://aws.amazon.com/ec2)

---

## 🌐 Live Demo

**배포 URL**: [http://3.36.178.61:5001](http://3.36.178.61:5001)

> AWS EC2(Ubuntu 24.04, t3.micro)에서 직접 운영 중입니다. Elastic IP로 고정 주소를 확보했고, systemd로 24시간 자동 실행 및 장애 시 자동 재시작이 구성되어 있습니다.

---

## 📌 프로젝트 개요 | Overview

FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 다분기 데이터(약 480,000건)를 기반으로, 약물별 이상반응 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다.

**핵심 특징:**
- Optuna 하이퍼파라미터 자동 탐색 + SHAP/LIME 설명 가능한 AI(XAI)
- **환자 단위(primaryid) GroupShuffleSplit으로 데이터 누수 방지**, 모델 신뢰도 및 확률 보정(Calibration) 검증
- MLflow 실험 추적 + Prophet 시계열 예측 + 분기별 자동 성능 모니터링 파이프라인
- K-Means 약물 클러스터링 + Co-medication 연관 분석 추천 시스템
- PubMed 논문 FAISS 벡터DB 임베딩 기반 RAG 약물 안전성 Q&A 챗봇
- PRR(Evans) + EBGM(FDA MGPS 베이지안) + MedDRA SOC 분류 3종 신호 탐지
- ICH E2B(R3) XML 자동 생성 + 21 CFR Part 11 전자서명
- **Docker 컨테이너화 검증 완료** + GitHub Actions CI/CD 파이프라인

---

## 🚀 주요 기능 | Key Features

### 📊 분석 & 예측
| 기능 | 설명 |
|------|------|
| Dashboard | FAERS 데이터 기반 이상반응 통계 시각화 (6개 차트) |
| AI 위험도 예측 | 약물·이상반응·나이·성별 입력 시 XGBoost 고위험/저위험 자동 판정 |
| SHAP / LIME XAI | SHAP(Global) + LIME(Local) 예측 근거 비교 시각화 |
| 분기별 트렌드 + 예측 | 2024 Q1~2025 Q1 분기별 추이 + Prophet 향후 2개 분기 예측 |

### 🧪 MLOps 파이프라인
| 기능 | 설명 |
|------|------|
| Optuna 탐색 | 9개 하이퍼파라미터 자동 탐색 (CV 3-fold) |
| MLflow 실험 추적 | run별 파라미터/메트릭 자동 기록·비교 |
| ML Dashboard | MLflow 실험 결과 웹 대시보드 (`/ml_dashboard`) |
| 자동 재학습 | 새 FAERS 분기 데이터 자동 다운로드 → 전처리 → 재학습 → MLflow 기록 |
| **데이터 누수 검증** | 환자(primaryid) 단위 GroupShuffleSplit, risk-rate 피처는 train만으로 계산 |
| **모델 비교 실험** | Logistic Regression / RandomForest / LightGBM / XGBoost 동일 split 비교 |
| **Calibration 검증** | Isotonic Regression으로 확률 보정, Brier Score 측정 |
| **Drift Monitoring** | 분기별 정적 모델 성능 추적, 재학습 필요성 정량 검증 |

### 💡 추천 시스템
| 기능 | 설명 |
|------|------|
| Drug Clustering | K-Means로 유사 부작용 프로필 약물 추천 (8개 클러스터) |
| Co-medication 분석 | 함께 복용된 약물 Top 10 + 중증 부작용 비율 분석 |

### 💊 약물 검색 & 정보
| 기능 | 설명 |
|------|------|
| Drug Lookup | 약품명/모양/색상으로 식약처+OpenFDA 정보 조회 + AI 안전성 리포트 PDF |
| Interaction Checker | FDA FAERS 기반 약물 병용 시 이상반응 위험 분석 |
| Drug Comparison | 두 약물 통계 나란히 비교 + AI 안전성 리포트 동시 생성 |
| Data Filter | 약물명·성별·나이·결과·국가 조건 필터링 |
| Dosage Calculator | CrCl·소아용량·BSA 항암제·mg/kg 표준용량 계산 |

### 🔬 RAG & AI 문헌 분석
| 기능 | 설명 |
|------|------|
| RAG 약물 Q&A | PubMed 30개 약물 1,689개 청크를 FAISS 벡터DB에 임베딩, 근거 기반 답변 |
| AI 안전성 리포트 | FDA FAERS + PubMed 5편 통합 6개 섹션 자동 생성 + PDF 다운로드 |
| 논문 검색 | PubMed API 논문 검색 + Claude AI 한국어 요약 |

### 📡 신호 탐지 & 규제 준수
| 기능 | 설명 |
|------|------|
| PRR 신호 탐지 | Evans 기준(PRR ≥ 2, n ≥ 3) 이상반응 신호 탐지 |
| EBGM 신호 탐지 | FDA MGPS 베이지안 알고리즘 근사 (EB05 ≥ 2 기준) |
| MedDRA SOC 분류 | System Organ Class 기반 부작용 체계 분류 |
| AE Manager | CTCAE 자동 등급·SAE 판정·15일 보고 마감·ICH E2B(R3) XML |
| 21 CFR Part 11 | SHA-256 전자서명·비밀번호 보호·이력 DB 보존 |
| Audit Trail | 모든 AE 데이터 생성/수정/삭제/열람 이력 자동 기록 |

---

## 🧠 ML/MLOps 파이프라인 | ML Pipeline

```
[FDA FAERS 다분기 데이터]
        ↓
[환자(primaryid) 단위 GroupShuffleSplit] ← 데이터 누수 방지
        ↓
[Train 데이터로만 risk-rate 피처 생성] ← 피처 누수 방지
  drug_risk_rate / reac_risk_rate / combo_risk_rate
        ↓
[Optuna 하이퍼파라미터 탐색]
  CV 3-fold → 최적 파라미터 탐색
        ↓
[모델 비교: LogReg / RandomForest / LightGBM / XGBoost]
        ↓
[XGBoost 학습 + Calibration]
  Isotonic Regression으로 확률 보정 (Brier Score 개선)
        ↓
[MLflow 실험 기록]
  파라미터 / 메트릭 / 모델 아티팩트 자동 저장
        ↓
[SHAP + LIME XAI]
  Global(SHAP) + Local(LIME) 예측 설명
        ↓
[분기별 Drift Monitoring]
  분기마다 성능 추적 → 재학습 필요성 정량 검증
```

---

## 🔍 데이터 누수 방지 | Data Leakage Prevention

FAERS 데이터는 동일 환자(`primaryid`)가 여러 약물·여러 이상반응으로 다중 행에 등장할 수 있습니다. 단순 `train_test_split`을 사용하면 같은 환자의 다른 레코드가 train과 test에 동시에 들어가 평가 지표가 실제보다 부풀려질 수 있습니다.

**적용한 방법:**
```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["primaryid"]))
```

추가로, `drug_risk_rate` / `reac_risk_rate` / `combo_risk_rate` 같은 위험률 피처는 **train 데이터로만 계산**한 뒤 test에 매핑하여(train에 없는 값은 global rate로 fallback) 피처 자체의 정보 누수도 차단했습니다. 코드 내에서 train/test 환자 ID 중복 여부를 `assert`로 검증하여 누수가 없음을 보장합니다.

**검증 결과:** 데이터 누수 제거 전후 성능을 비교한 결과, 정확도가 다소 낮아졌으나(아래 Model Performance 참고) 이는 실제 모델 성능을 정직하게 반영한 결과이며, 향후 피처 엔지니어링 개선의 출발점으로 활용하고 있습니다.

---

## ⚖️ 모델 비교 실험 | Model Comparison

동일한 환자 단위 split(GroupShuffleSplit, primaryid 기준)으로 4개 모델을 공정하게 비교했습니다.

| Model | Accuracy | F1 (risk) | Recall (risk) | Precision (risk) |
|---|---|---|---|---|
| RandomForest | 0.494 | **0.450** | **0.580** | 0.367 |
| XGBoost | 0.520 | 0.410 | 0.468 | 0.365 |
| LightGBM | 0.521 | 0.407 | 0.461 | 0.364 |
| LogisticRegression | **0.549** | 0.351 | 0.342 | 0.360 |

**모델 선택 근거:** RandomForest가 F1과 Recall에서 가장 높지만, 4개 모델 간 Precision 차이가 거의 없어(0.36 안팎) 현재 피처셋의 한계로 판단했습니다. XGBoost를 최종 모델로 채택한 이유는 (1) Optuna 기반 하이퍼파라미터 자동 튜닝 인프라가 이미 구축되어 있고, (2) SHAP 기반 설명 가능성 모듈과의 통합이 검증되어 있으며, (3) Accuracy와 F1의 균형이 RandomForest 대비 합리적이기 때문입니다. 다만 의료 스크리닝 목적상 Recall이 중요한 경우 RandomForest가 더 적합할 수 있다는 트레이드오프를 인지하고 있습니다.

---

## 🎯 Calibration 검증 | Probability Calibration

의료 AI에서 "위험도 87%"라는 출력이 실제로 87% 확률을 의미하는지는 모델 정확도와는 별개의 문제입니다. Isotonic Regression으로 확률을 보정하고 Brier Score로 검증했습니다.

| | Brier Score (낮을수록 좋음) |
|---|---|
| Base XGBoost | 0.2954 |
| Calibrated (Isotonic) | **0.2588** (12.4% 개선) |

Calibration은 환자 단위로 분리된 별도의 held-out calibration set(학습/테스트와 모두 비중복)으로 fit하여, calibration 과정 자체에서도 데이터 누수가 없도록 구성했습니다 (`sklearn.frozen.FrozenEstimator` 사용).

---

## 📉 Drift Monitoring | 분기별 성능 추적

2024Q1 데이터로만 학습한 모델을 이후 분기(Q2~2025Q1)에 그대로 적용해 성능 변화를 추적했습니다.

| Quarter | F1 | Recall | Precision |
|---|---|---|---|
| 2024Q1 (학습 분기) | 0.752 | 0.805 | 0.705 |
| 2024Q2 | 0.383 | 0.398 | 0.368 |
| 2024Q3 | 0.376 | 0.397 | 0.357 |
| 2025Q1 | 0.377 | 0.397 | 0.359 |

학습 분기 바로 다음 분기부터 F1이 절반 가까이 급락하는 패턴이 관찰되었습니다. 이는 LabelEncoder 기반 약물/이상반응 피처가 학습 시점에 없던 신규 코드를 처리하지 못하는 구조적 한계와 실제 데이터 분포 변화(drift)가 결합된 결과로 추정되며, **주기적 재학습 파이프라인(MLflow + 분기별 자동 재학습)의 필요성을 정량적으로 뒷받침하는 근거**로 활용하고 있습니다.

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend    : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI      : XGBoost, Optuna, SHAP, LIME, LightGBM, RandomForest (비교 실험)
MLOps      : MLflow (실험 추적), Prophet (시계열 예측), scikit-learn
검증        : GroupShuffleSplit(데이터 누수 방지), Calibration(Isotonic), Drift Monitoring
추천 시스템 : K-Means 클러스터링, Co-medication 연관 분석
RAG        : LangChain, FAISS, sentence-transformers, llama3.2 (Ollama)
Data       : FDA FAERS 2024 Q1~2025 Q1 (~480,000건)
External   : 식약처 의약품안전나라 OpenAPI, OpenFDA, PubMed E-utilities API
Viz        : Plotly, Chart.js, NetworkX
DB         : SQLite (mlflow.db 포함)
Report     : ReportLab (PDF), ICH E2B(R3) XML
Compliance : 21 CFR Part 11 전자서명, Audit Trail, ICH E2B(R3)
Frontend   : Jinja2 Templates, Vanilla JS, 반응형 CSS, PWA
Container  : Docker (빌드·실행 검증 완료, torch CPU 전용 빌드로 경량화)
Infra      : AWS EC2 (t3.micro, Ubuntu 24.04, 서울 리전) + Elastic IP + systemd
CI/CD      : GitHub Actions (push/PR 시 pytest 자동 실행)
Test       : pytest (92개 유닛테스트)
```

---

## ☁️ Infrastructure (AWS EC2)

- **Compute**: AWS EC2 t3.micro, Ubuntu 24.04 LTS, 서울 리전(ap-northeast-2)
- **고정 IP**: Elastic IP 적용 — 인스턴스 재시작에도 URL 불변
- **배포 방식**: SSH 기반 직접 배포, Python venv 가상환경 + systemd 서비스로 24시간 운영 및 자동 재시작
- **보안 그룹**: SSH(22), HTTP(80), HTTPS(443), Flask(5001) 포트 커스텀 관리
- **스토리지**: EBS 30GB(gp3), 스왑 메모리 2GB 추가 구성
- **컨테이너화**: Dockerfile 작성 및 빌드·실행 검증 완료 (gunicorn 기반 프로덕션 서버)
- **마이그레이션**: Railway(PaaS) → AWS EC2(IaaS) 전환을 통해 서버 인프라 직접 구축·운영 경험 확보

---

## 📁 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   └── routes/
│       ├── analysis.py    # PRR + EBGM + SHAP + LIME + Prophet
│       ├── recommend.py   # K-Means 클러스터링 + Co-medication 분석
│       ├── ml_dashboard.py # MLflow 실험 결과 대시보드
│       ├── drug.py
│       ├── ae.py
│       ├── auth.py
│       ├── vision.py
│       ├── literature.py
│       └── rag.py
├── ml/
│   ├── train_model_optuna.py    # Optuna + MLflow 학습 (데이터 누수 방지 적용)
│   ├── compare_models.py        # 모델 비교 실험
│   ├── calibration_check.py     # Calibration 검증
│   ├── drift_monitoring.py      # 분기별 Drift 모니터링
│   ├── retrain_pipeline.py      # 분기별 자동 재학습
│   ├── model.pkl
│   ├── best_params.json
│   ├── model_comparison.json/.md
│   ├── calibration_report.json / calibration_curve.png
│   └── drift_report.json / drift_trend.png
├── data/
│   └── processed/processed_faers.csv
├── Dockerfile
├── .github/workflows/tests.yml
├── mlflow.db
├── config.py
├── run.py
└── README.md
```

---

## ⚙️ 설치 및 실행 | Installation & Run

```bash
# 1. 저장소 복제
git clone https://github.com/leesihwan21/pharma-risk-analyzer.git
cd pharma-risk-analyzer

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (.env)
SECRET_KEY=your-secret-key
MFDS_API_KEY=your-mfds-api-key
ANTHROPIC_API_KEY=your-api-key

# 5. ML 모델 학습 (데이터 누수 방지 적용된 버전)
python ml/train_model_optuna.py

# 6. (선택) 모델 비교 / Calibration / Drift 검증
python ml/compare_models.py
python ml/calibration_check.py
python ml/drift_monitoring.py

# 7. 서버 실행
python run.py
# → http://127.0.0.1:5001

# 8. (선택) Docker로 실행
docker build -t pharma-risk-analyzer .
docker run -d -p 5001:5001 --env-file .env pharma-risk-analyzer

# 9. MLflow UI (별도 터미널)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
# → http://127.0.0.1:5002
```

---

## 📊 ML 모델 성능 | Model Performance

| 지표 | 값 | 설명 |
|------|-----|------|
| Accuracy | 52.2% | 환자 단위 split 적용 후 정직한 성능 |
| F1 (위험) | 0.407 | 위험 클래스 F1 |
| Recall (위험) | 0.468 | 위험 케이스 탐지율 |
| Precision (위험) | 0.365 | 위험 예측 정확도 |
| Brier Score (Calibrated) | 0.2588 | 확률 보정 후 (12.4% 개선) |

> **데이터 누수 검증으로 인한 정직한 재평가**: 초기 단순 split 기준으로는 Accuracy 69.3%였으나, 환자(primaryid) 단위 GroupShuffleSplit과 피처 누수 제거를 적용한 결과 52.2%로 재산정되었습니다. 이는 모델 성능을 부풀리지 않고 정직하게 검증한 결과이며, Drift Monitoring 결과와 함께 향후 피처 엔지니어링 및 재학습 전략 수립의 근거 자료로 사용하고 있습니다.

---

## 📁 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 약물 이상반응 자발적 보고 데이터
- **식약처 이상반응**: 연도별(2019~2024) 국내 이상반응 보고 통계
- **식약처 의약품안전나라 OpenAPI**: 공공데이터포털(data.go.kr)
- **OpenFDA Drug Label API**: FDA 공식 약물 설명서
- **PubMed E-utilities API**: NCBI 논문 검색 및 초록 수집 (무료)

---

## ⚠️ 면책조항 | Disclaimer

본 솔루션은 **연구·교육·포트폴리오 목적**으로 제작되었으며, 실제 임상 처방 결정에 사용해서는 안 됩니다.

---

## 👤 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
