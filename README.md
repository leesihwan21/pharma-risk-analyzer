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

> AWS EC2(Ubuntu 24.04, t3.micro)에서 gunicorn + systemd 기반으로 직접 운영 중입니다. Elastic IP로 고정 주소를 확보했고, IAM 역할 기반 권한 관리와 CloudWatch를 통해 애플리케이션 로그·CPU·메모리·디스크 메트릭을 실시간으로 모니터링하고 있습니다.

---

## 📌 프로젝트 개요 | Overview

FDA FAERS(Adverse Event Reporting System) 2024 Q1 ~ 2025 Q1 분기별 데이터(약 480,000건)를 기반으로, 약물별 부작용 발생 패턴을 분석하고 XGBoost 머신러닝으로 위험도를 예측하는 웹 애플리케이션입니다.

**핵심 특징:**
- Optuna 하이퍼파라미터 자동 탐색 + SHAP/LIME 설명 가능한 AI(XAI)
- **환자 단위(primaryid) GroupShuffleSplit으로 데이터 누수 방지**, 모델 신뢰성 및 확률 보정(Calibration) 검증
- MLflow 실험 추적 + Prophet 시계열 예측 + 분기별 자동 성능 모니터링 파이프라인
- K-Means 약물 클러스터링 + Co-medication 연관 분석 추천 시스템
- PubMed 논문 FAISS 벡터DB 임베딩 기반 RAG 약물 안전성 Q&A 챗봇
- YOLOv8 + OCR 기반 알약 이미지 인식 (식약처 낱알식별 25,322건 로컬 캐싱)
- PRR(Evans) + EBGM(FDA MGPS 베이지안) + MedDRA SOC 분류 3종 신호 탐지
- ICH E2B(R3) XML 자동 생성 + 21 CFR Part 11 전자서명
- **Docker 컨테이너 검증 완료** + GitHub Actions CI/CD 파이프라인
- AWS EC2 gunicorn + systemd 운영, **AWS RDS PostgreSQL 관리형 DB**, IAM 역할 기반 권한 관리, CloudWatch 로그·메트릭 모니터링

---

## 🎯 주요 기능 | Key Features

### 📊 분석 & 예측
| 기능 | 설명 |
|------|------|
| Dashboard | FAERS 데이터 기반 부작용 통계 시각화 (6개 차트) |
| AI 위험도 예측 | 약물·부작용·나이·성별 입력 시 XGBoost 기반 위험도 자동 판정 |
| SHAP / LIME XAI | SHAP(Global) + LIME(Local) 예측 근거 비교 시각화 |
| 분기별 트렌드 + 예측 | 2024 Q1~2025 Q1 분기별 추이 + Prophet 이후 2개 분기 예측 |

### 🤖 MLOps 파이프라인
| 기능 | 설명 |
|------|------|
| Optuna 탐색 | 9개 하이퍼파라미터 자동 탐색 (CV 3-fold) |
| MLflow 실험 추적 | run별 파라미터/메트릭 자동 기록·비교 |
| ML Dashboard | MLflow 실험 결과 웹 대시보드 (`/ml_dashboard`) |
| 자동 재학습 | 신규 FAERS 분기 데이터 자동 다운로드 → 전처리 → 재학습 시 MLflow 기록 |
| **데이터 누수 검증** | 환자(primaryid) 단위 GroupShuffleSplit, risk-rate 피처는 train만으로 계산 |
| **모델 비교 실험** | Logistic Regression / RandomForest / LightGBM / XGBoost 동일 split 비교 |
| **Calibration 검증** | Isotonic Regression으로 확률 보정, Brier Score 측정 |
| **Drift Monitoring** | 분기별 고정 모델 성능 추적, 재학습 필요성 정량 검증 |

### 💡 추천 시스템
| 기능 | 설명 |
|------|------|
| Drug Clustering | K-Means로 유사 부작용 프로필 약물 추천 (8개 클러스터) |
| Co-medication 분석 | 함께 복용된 약물 Top 10 + 중증 부작용 비율 분석 |

### 💊 약물 검색 & 정보
| 기능 | 설명 |
|------|------|
| Drug Lookup | 제품명·모양·색상으로 식약처/OpenFDA 정보 조회 + AI 안전성 리포트 PDF |
| **알약 이미지 인식** | YOLOv8 알약 탐지 + OCR 식별문자 인식 + 식약처 낱알식별 로컬 DB 캐시(25,322건) + e약은요(DUR) 연동 효능·용법·부작용 조회 |
| Interaction Checker | FDA FAERS 기반 약물 병용 시 부작용 위험 분석 |
| Drug Comparison | 두 약물 통계 나란히 비교 + AI 안전성 리포트 동시 생성 |
| Data Filter | 약물명·성별·나이·국가 등 조건 필터링 |
| Dosage Calculator | CrCl·체표면적(BSA) 기반 mg/kg 권장 용량 계산 |
| **한국 DUR 병용금기 검증** | FAERS 기반 위험 약물 조합을 한국 식약처 DUR(병용금기) 데이터베이스와 실시간 대조 검증 |

### 📚 RAG & AI 문헌 분석
| 기능 | 설명 |
|------|------|
| RAG 약물 Q&A | PubMed 30개 약물 1,689개 청크를 FAISS 벡터DB에 임베딩, 근거 기반 답변 |
| AI 안전성 리포트 | FDA FAERS + PubMed 5건 통합 6개 섹션 자동 생성 + PDF 다운로드 |
| 논문 검색 | PubMed API 논문 검색 + Claude AI 한국어 요약 |

### 🚨 신호 탐지 & 규제 준수
| 기능 | 설명 |
|------|------|
| PRR 신호 탐지 | Evans 기준(PRR ≥ 2, n ≥ 3) 부작용 신호 탐지 |
| EBGM 신호 탐지 | FDA MGPS 베이지안 알고리즘 근사 (EB05 ≥ 2 기준) |
| MedDRA SOC 분류 | System Organ Class 기반 부작용 체계 분류 |
| AE Manager | CTCAE 자동 등급·SAE 판정·15일 보고 마감 관리·ICH E2B(R3) XML |
| 21 CFR Part 11 | SHA-256 전자서명·비밀번호 보호·이력 DB 보존 |
| Audit Trail | 모든 AE 데이터 생성/수정/삭제/열람 이력 자동 기록 |

---

## 🔬 ML/MLOps 파이프라인 | ML Pipeline

```
[FDA FAERS 분기 데이터]
        ↓ [환자(primaryid) 단위 GroupShuffleSplit] → 데이터 누수 방지
        ↓ [Train 데이터로만 risk-rate 피처 생성] → 피처 누수 방지
  drug_risk_rate / reac_risk_rate / combo_risk_rate
        ↓ [Optuna 하이퍼파라미터 탐색]
  CV 3-fold → 최적 파라미터 탐색
        ↓ [모델 비교: LogReg / RandomForest / LightGBM / XGBoost]
        ↓ [XGBoost 학습 + Calibration]
  Isotonic Regression으로 확률 보정 (Brier Score 개선)
        ↓ [MLflow 실험 기록]
  파라미터 / 메트릭 / 모델 아티팩트 자동 저장
        ↓ [SHAP + LIME XAI]
  Global(SHAP) + Local(LIME) 예측 설명
        ↓ [분기별 Drift Monitoring]
  분기마다 성능 추적 → 재학습 필요성 정량 검증
```

---

## 🔍 데이터 누수 방지 | Data Leakage Prevention

FAERS 데이터는 동일 환자(`primaryid`)가 여러 약물·여러 부작용으로 중복 행에 등장할 수 있습니다. 단순 `train_test_split`을 사용하면 같은 환자의 다른 레코드가 train과 test에 동시에 들어가 평가 지표가 실제보다 부풀려질 수 있습니다.

**적용한 방법:**
```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["primaryid"]))
```

추가로, `drug_risk_rate` / `reac_risk_rate` / `combo_risk_rate` 같은 위험률 피처를 **train 데이터로만 계산**한 뒤 test에 매핑하여(train에 없는 값은 global rate로 fallback) 피처 자체의 정보 누수도 차단했습니다. 코드 내에는 train/test 환자 ID 중복 여부를 `assert`로 검증하는 안전장치가 있어 누수가 없음을 보장합니다.

**검증 결과:** 데이터 누수 제거 전후 성능을 비교한 결과 정확도가 다소 낮아졌지만(아래 Model Performance 참고), 이는 실제 모델 성능을 정직하게 반영한 결과이며, 이후 피처 보완 작업의 출발점으로 활용하고 있습니다.

---

## 🏆 모델 비교 실험 | Model Comparison

동일한 환자 단위 split(GroupShuffleSplit, primaryid 기준)으로 4개 모델을 공정하게 비교했습니다. (Bayesian 스무딩 적용 + 원시 LabelEncoder ID 제거 후 수치)

| Model | Accuracy | F1 (risk) | Recall (risk) | Precision (risk) |
|---|---|---|---|---|
| LogisticRegression | **0.527** | 0.406 | 0.453 | **0.368** |
| RandomForest | 0.508 | **0.420** | **0.500** | 0.362 |
| LightGBM | 0.519 | 0.409 | 0.468 | 0.364 |
| XGBoost | 0.520 | 0.405 | 0.458 | 0.363 |

**Target Encoding 개선 효과:** 원래 피처셋에는 `drug_encoded`/`reac_encoded`(의미 없는 LabelEncoder 정수 ID)가 risk-rate 피처와 함께 그대로 모델 입력으로 들어가 있었습니다. 이 원시 ID를 제거하고 risk-rate 피처에 표본 수 기반 Bayesian 스무딩(`(count·mean + k·global_rate) / (count + k)`, k=20)을 적용한 결과, **LogisticRegression의 F1이 0.351 → 0.406으로 크게 개선**되었습니다. 선형 모델은 의미 없는 정수 인코딩에 특히 민감하기 때문에, 이 개선이 원시 ID 제거가 실제로 유효한 수정이었음을 보여주는 근거가 됩니다. 트리 모델들은 원래도 무의미한 분기를 일부 무시할 수 있어 변화가 상대적으로 작았습니다.

**모델 선택 근거:** RandomForest가 F1(0.420)과 Recall(0.500)에서 가장 높게 나타납니다. 단순 지표만 보면 RandomForest가 더 좋은 선택처럼 보이지만, XGBoost를 최종 모델로 채택한 이유는 다음과 같습니다.

1. **SHAP 설명 가능성과의 통합**: XGBoost는 SHAP TreeExplainer와 호환성이 가장 안정적이며, 이미 Global/Local 설명 파이프라인이 구축되어 있습니다.
2. **안정적인 Optuna 최적화 워크플로우**: 9개 하이퍼파라미터에 대한 베이지안 탐색 파이프라인이 XGBoost 기준으로 구축되어 있어, 추가 데이터 확보 시 즉시 재학습이 가능합니다.
3. **빠른 재학습 파이프라인**: Drift Monitoring 결과(아래 참고)가 보여주듯 분기별 재학습이 필수적인데, XGBoost는 RandomForest보다 학습 속도가 빠르고 증분 학습 확장에 유리합니다.
4. **프로덕션 배포 확장성**: 모델 직렬화 크기, 추론 속도, gunicorn 환경에서의 메모리 사용량 면에서 XGBoost가 더 가볍습니다.

즉 "어떤 모델이 가장 높은 점수를 받았는가"가 아니라 "어떤 모델이 SHAP·Optuna·재학습 자동화까지 포함한 전체 MLOps 파이프라인에 가장 잘 맞는가"를 기준으로 선택했습니다. 다만 의료 스크리닝처럼 위험 케이스를 놓치지 않는 것이 최우선인 환경에서는 Recall이 높은 RandomForest가 더 적합할 수 있다는 트레이드오프를 인지하고 있으며, 이는 추후 앙상블(XGBoost + RandomForest) 검토 과제로 남겨두고 있습니다.

---

## 🎚️ Calibration 검증 | Probability Calibration

의료 AI에서 "위험도 87%"라는 출력이 실제로 87% 확률로 발생하는지는 모델 정확도와는 별개의 문제입니다. Isotonic Regression으로 확률을 보정하고 Brier Score로 검증했습니다.

| | Brier Score (낮을수록 좋음) |
|---|---|
| Base XGBoost | 0.2945 |
| Calibrated (Isotonic) | **0.2587** (12.1% 개선) |

Calibration은 환자 단위로 분리된 별도의 held-out calibration set(학습/테스트와 모두 비중복)으로 fit하여, calibration 과정 자체에서도 데이터 누수가 없도록 구성했습니다 (`sklearn.frozen.FrozenEstimator` 사용). Target Encoding 스무딩 적용 전후로 Brier Score는 거의 변화가 없었는데(0.2954→0.2945), 이는 자연스러운 결과입니다 — 스무딩은 피처의 노이즈를 줄이는 것이고 Calibration은 모델이 이미 내놓은 확률 출력을 보정하는 것이라, 두 개선이 서로 다른 문제를 다루기 때문입니다.

---

## 📉 Drift Monitoring | 분기별 성능 추적

2024Q1 데이터로만 학습한 모델을 이후 분기(Q2~2025Q1)에 그대로 적용했을 때 성능 변화를 추적했습니다.

**평가 누수 버그 발견 및 수정:** 처음 측정했을 때는 학습 분기(2024Q1)의 F1이 0.752로 다른 분기(0.38 전후)보다 훨씬 높게 나와, "분기가 바뀌면 성능이 절반 가까이 추락한다"는 극심한 드리프트로 보였습니다. 그런데 이 수치를 다시 들여다보니, 학습 분기를 평가할 때 **학습에 실제로 쓰인 80% 데이터까지 포함된 전체 Q1**으로 평가하고 있었다는 걸 발견했습니다 — 모델이 이미 외운 데이터로 시험을 본 셈이라 Q1 점수만 부풀려져 있었던 것입니다. 학습 분기도 학습에 전혀 쓰이지 않은 held-out 20%로만 평가하도록 수정한 결과는 다음과 같습니다.

| Quarter | F1 | Recall | Precision |
|---|---|---|---|
| 2024Q1 (학습 분기, held-out) | 0.382 | 0.402 | 0.363 |
| 2024Q2 | 0.392 | 0.418 | 0.370 |
| 2024Q3 | 0.384 | 0.416 | 0.357 |
| 2025Q1 | 0.383 | 0.411 | 0.358 |

수정 후에는 모든 분기가 F1 0.38~0.39 사이에 머물러, **실질적인 드리프트가 거의 관찰되지 않습니다.** 즉 처음 발견했던 "심각한 드리프트"는 대부분 모델의 시간적 불안정성이 아니라 평가 방식 자체의 버그였다는 결론입니다. 다만 절대적인 F1 자체가 0.38 수준으로 낮은 점은 여전한 한계이며, 이는 드리프트 문제가 아니라 피처/모델 구조 자체의 개선이 필요함을 시사합니다 (아래 "향후 개선 계획" 참고).

이 경험은 모델 성능 수치 자체보다, **평가 파이프라인에 숨어 있는 버그를 의심하고 검증하는 과정**이 더 중요하다는 것을 보여줍니다. 처음에 봤던 "급격한 드리프트" 그래프를 그대로 믿고 결론을 냈다면, 실제로는 존재하지 않는 문제를 해결하려고 시간을 쓸 뻔했습니다.

---

## 📚 RAG 평가 | RAG Evaluation

PubMed 기반 RAG 챗봇의 응답 품질을 정량적으로 평가했습니다. RAGAS 라이브러리는 의존성 충돌(`langchain_community`의 vertexai 모듈 누락)로 직접 실행이 불가능해, 동일한 평가 방법(Faithfulness, Answer Relevancy, Context Precision)을 LLM judge(llama3.1:8b) 호출로 직접 구현했습니다. 자기평가 영향을 줄이기 위해 답변 생성(llama3.2)과 평가(llama3.1:8b) 모델을 분리했습니다.

| Metric | 평균 점수 | 의미 |
|---|---|---|
| Faithfulness | 0.62 | 답변이 검색된 문헌에 얼마나 근거하는지 |
| Answer Relevancy | 0.82 | 답변이 질문에 얼마나 직접적으로 답하는지 |
| Context Precision | 0.34 | 검색된 청크가 질문에 실제로 유용한지 |

**진단:** Faithfulness와 Answer Relevancy는 양호한 수준이지만, Context Precision은 상대적으로 낮게(5개 질문 중 1개는 0.0) 측정되었습니다. 이는 생성(Generation) 단계보다 검색(Retrieval) 단계가 RAG 파이프라인의 병목임을 시사합니다. 5개 평가 질문 중 한 질문에서 Faithfulness가 0.2로 가장 낮게 측정되었는데, 검색된 컨텍스트 밖의 내용(계산식·제제 관련 기술)이 답변에 포함되었기 때문으로 보입니다. 이후 임베딩 모델 교체(`all-MiniLM-L6-v2` 대신 의학 전문 특화 임베딩) 또는 청크 분할 전략 개선이 우선 과제로 식별되었습니다.

---

## 🛠️ 기술 스택 | Tech Stack

```
Backend    : Flask 3.1, SQLAlchemy, Flask-Login, Flask-Limiter, Flask-Caching
ML/AI      : XGBoost, Optuna, SHAP, LIME, LightGBM, RandomForest (비교 실험)
MLOps      : MLflow (실험 추적), Prophet (시계열 예측), scikit-learn
검증       : GroupShuffleSplit(데이터 누수 방지), Smoothed Target Encoding(과적합 방지), Calibration(Isotonic), Drift Monitoring(평가 누수 버그 수정), RAG 평가(Faithfulness/Relevancy/Context Precision)
Vision     : YOLOv8 (알약 탐지), EasyOCR (식별문자 인식)
추천 시스템 : K-Means 클러스터링, Co-medication 연관 분석
RAG        : LangChain, FAISS, sentence-transformers, llama3.2 (Ollama)
Data       : FDA FAERS 2024 Q1~2025 Q1 (~480,000건), 식약처 낱알식별 정보 (~25,322건)
External   : 식약처 의약품안전나라 OpenAPI(낱알식별/DUR), OpenFDA, PubMed E-utilities API
Viz        : Plotly, Chart.js, NetworkX
DB         : PostgreSQL (AWS RDS, 애플리케이션 메인 DB) + SQLite (mlflow.db, pill_identity.db)
Report     : ReportLab (PDF), ICH E2B(R3) XML
Compliance : 21 CFR Part 11 전자서명, Audit Trail, ICH E2B(R3)
Frontend   : Jinja2 Templates, Vanilla JS, 반응형 CSS, PWA
Container  : Docker (빌드·실행 검증 완료, torch CPU 전용 빌드로 경량화)
Infra      : AWS EC2 (t3.micro, Ubuntu 24.04, 서울 리전) + AWS RDS PostgreSQL + Elastic IP + gunicorn + systemd + IAM + CloudWatch
CI/CD      : GitHub Actions (push/PR 시 pytest 자동 실행)
Test       : pytest (92개 자동 테스트)
```

---

## ☁️ Infrastructure (AWS EC2)

- **Compute**: AWS EC2 t3.micro, Ubuntu 24.04 LTS, 서울 리전(ap-northeast-2)
- **고정 IP**: Elastic IP 적용 → 인스턴스 재시작에도 URL 불변
- **배포 방식**: gunicorn(3 workers) + systemd 서비스(`pharma.service`)로 24시간 운영 및 장애 시 자동 재시작. `EnvironmentFile`로 `.env` 환경변수 안전하게 로드
- **데이터베이스**: AWS RDS PostgreSQL(db.t4g.micro, 프리 티어)로 메인 애플리케이션 DB를 SQLite에서 전환. 퍼블릭 액세스를 비활성화하고 EC2 인스턴스를 보안 그룹에 직접 연결해 같은 VPC 내부에서만 접근 가능하도록 구성. 로컬 관리(pgAdmin)는 EC2를 경유하는 SSH 터널로만 연결해 DB를 인터넷에 직접 노출하지 않음
- **보안**: Flask 개발 서버(`debug=True`, 퍼블릭 노출) 운영 방식을 gunicorn 프로덕션 구성으로 전환해 Werkzeug 디버거 원격 코드 실행 위험 제거
- **보안 그룹**: SSH(22), HTTP(80), HTTPS(443), Flask(5001) 포트 커스텀 관리
- **스토리지**: EBS 30GB(gp3), 스왑 메모리 2GB 추가 구성
- **IAM 역할 기반 권한 관리**: `CloudWatchAgentServerPolicy` + `AmazonS3FullAccess`를 포함한 전용 IAM 역할(`pharma-risk-analyzer-ec2-role`)을 EC2 인스턴스에 연결
- **모니터링**: CloudWatch 에이전트로 gunicorn 애플리케이션 로그(`/pharma-risk-analyzer/gunicorn` 로그 그룹) 및 CPU·메모리·디스크 메트릭(CWAgent 네임스페이스)을 실시간 수집
- **컨테이너**: Dockerfile 작성 및 빌드·실행 검증 완료 (gunicorn 기반 프로덕션 서버)
- **마이그레이션**: Railway(PaaS) → AWS EC2(IaaS) 전환, SQLite → AWS RDS PostgreSQL 전환을 통해 서버 인프라와 데이터베이스를 직접 구축·운영하는 경험 확보

---

## 📁 프로젝트 구조 | Project Structure

```
pharma-risk-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py
│   └── routes/
│       ├── analysis.py     # PRR + EBGM + SHAP + LIME + Prophet
│       ├── recommend.py    # K-Means 클러스터링 + Co-medication 분석
│       ├── ml_dashboard.py # MLflow 실험 결과 웹 대시보드
│       ├── drug.py
│       ├── ae.py
│       ├── auth.py
│       ├── vision.py       # 알약 이미지 인식 (YOLOv8 + OCR + 식약처 API)
│       ├── literature.py
│       └── rag.py
├── ml/
│   ├── train_model_optuna.py    # Optuna + MLflow 학습 (데이터 누수 방지 적용)
│   ├── compare_models.py        # 모델 비교 실험
│   ├── calibration_check.py     # Calibration 검증
│   ├── drift_monitoring.py      # 분기별 Drift 모니터링
│   ├── rag_evaluation.py        # RAG 평가 (Faithfulness/Relevancy/Context Precision)
│   ├── retrain_pipeline.py      # 분기별 자동 재학습
│   ├── model.pkl
│   ├── best_params.json
│   ├── model_comparison.json/.md
│   ├── calibration_report.json / calibration_curve.png
│   └── drift_report.json / drift_trend.png
├── data/
│   ├── pill_identity.db          # 식약처 낱알식별 정보 로컬 캐시 (25,322건)
│   └── processed/processed_faers.csv
├── build_pill_db.py               # 낱알식별 로컬 DB 빌드 스크립트
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
DATABASE_URL=postgresql://user:password@host:5432/dbname  # 미설정 시 SQLite로 자동 폴백
MFDS_API_KEY=your-mfds-api-key
ANTHROPIC_API_KEY=your-api-key

# 5. ML 모델 학습 (데이터 누수 방지 적용 버전)
python ml/train_model_optuna.py

# 6. (선택) 모델 비교 / Calibration / Drift / RAG 평가
python ml/compare_models.py
python ml/calibration_check.py
python ml/drift_monitoring.py
python ml/rag_evaluation.py

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
| Accuracy | 51.9% | 환자 단위 split 적용 후 정직한 성능 |
| F1 (위험) | 0.412 | 위험 클래스 F1 |
| Recall (위험) | 0.473 | 위험 케이스 탐지율 |
| Precision (위험) | 0.365 | 위험 예측 정확도 |
| Brier Score (Calibrated) | 0.2587 | 확률 보정 후 (12.1% 개선) |

> **데이터 누수 검증으로 인한 정직한 성능 평가**: 초기 단순 split 기준으로는 Accuracy 69.3%였으나, 환자(primaryid) 단위 GroupShuffleSplit과 피처 누수 제거를 적용한 결과 52% 수준으로 재산정되었습니다. 이후 risk-rate 피처에 Bayesian 스무딩을 적용하고 의미 없는 원시 LabelEncoder ID를 모델 입력에서 제거한 결과(Target Encoding 개선), F1이 0.407 → 0.412, Recall이 0.468 → 0.473으로 소폭 개선되었습니다. 이는 모델 성능을 부풀리지 않고 정직하게 검증한 결과이며, Drift Monitoring 결과(평가 누수 버그 수정 후 분기별 성능이 안정적임을 확인)와 함께 다음 단계 개선 방향을 수립하는 근거로 사용하고 있습니다.

---

## 📂 데이터 출처 | Data Sources

- **FDA FAERS 2024 Q1 ~ 2025 Q1**: FDA 공식 약물 이상반응 자발적 보고 데이터
- **식약처 이상반응**: 연도별(2019~2024) 국내 이상반응 보고 통계
- **식약처 의약품안전나라 OpenAPI**: 공공데이터포털(data.go.kr) — 낱알식별, DUR(e약은요)
- **식약처 DUR(의약품안전사용서비스) 병용금기 API**: 공공데이터포털(data.go.kr) — 한국 병용금기 기준 실시간 검증
- **OpenFDA Drug Label API**: FDA 공식 약물 설명서
- **PubMed E-utilities API**: NCBI 논문 검색 및 초록 수집 (무료)

---

## 📝 개발 배경 | Background

### 왜 이 프로젝트인가

임상약학 석사 과정(아주대학교)에서 약물 안전성 데이터를 다루며, 발생한 이상반응 보고 데이터가 실제로는 충분히 활용되지 못하고 있다는 점에 주목했습니다. FDA FAERS처럼 접근 가능한 실제 약물감시(pharmacovigilance) 데이터조차, 이를 머신러닝으로 분석하고 규제 기준(PRR/EBGM, ICH E2B, 21 CFR Part 11)까지 반영하는 도구는 흔하지 않습니다.

AI 개발 교육 과정(MBC아카데미)을 통해 쌓은 머신러닝·앱 개발 역량과 임상약학 백그라운드의 도메인 지식을 결합해, "약물 안전성 신호를 조기에 발견하고 그 판단 근거를 설명할 수 있는" 시스템을 만드는 것을 목표로 설정했습니다.

### 기획 의도

1. **단순 통계 시각화를 넘어선 예측**: FAERS 데이터를 보여주는 것에 머물지 않고, XGBoost로 위험도를 예측하고 SHAP/LIME으로 그 이유를 설명하는 것까지 구현
2. **규제 기준 반영**: 포트폴리오용 토이 프로젝트가 아니라, PRR/EBGM 같은 FDA/EMA 실제 신호 탐지 지표와 ICH E2B(R3), 21 CFR Part 11 같은 실제 제약업계 규제 기준을 적용
3. **모델을 신뢰할 수 있는가까지 직접 검증**: 모델을 만들고 끝내는 것이 아니라, 데이터 누수 검증·Calibration·Drift Monitoring을 통해 "이 모델을 실제로 믿을 수 있는가"를 직접 검증하는 과정도 프로젝트에 포함

### 개발 과정에서의 전환점

초기에는 Accuracy 69%라는 결과에 만족할 수도 있었지만, FAERS 데이터의 구조(동일 환자가 여러 레코드에 등장)를 고려하면 이 수치를 그대로 신뢰할 수 없다는 점을 인식하고 데이터 누수 검증을 진행했습니다. 그 결과 실제 성능은 52%로 재산정되었고, 이는 이후 "왜 분기별 재학습이 필요한가"를 보여주는 Drift Monitoring 실험으로 이어지는 계기가 되었습니다. 이 과정 자체가 모델 성능 수치보다 더 중요한 배움이라고 생각하여 README에도 정직하게 기록했습니다.

---

## 🎓 Key Lessons

Initially the model achieved 69% accuracy. However, after discovering patient-level leakage within FAERS reports — the same patient appearing in both train and test sets — the evaluation pipeline was redesigned using `GroupShuffleSplit` on patient ID (`primaryid`), and risk-rate features were recomputed using only the training set.

The resulting performance decreased to 52%, but became significantly more reliable and realistic. This experience highlighted the importance of **data validation over raw model metrics** — a lower, honest number is more valuable than a higher, leaked one. The same validation mindset was extended further: Calibration testing revealed that raw model probabilities did not match real-world outcome rates (improved via isotonic regression), and Drift Monitoring revealed that a model trained on a single quarter loses roughly half its F1 score by the very next quarter — providing quantitative justification for the quarterly retraining pipeline already built into this project.

---

## ⚠️ 면책조항 | Disclaimer

본 도구는 연구·교육·포트폴리오 목적으로 제작되었으며, 실제 임상 처방 결정에 사용해서는 안 됩니다.

---

## 👤 개발자 | Developer

**이시환 (Sihwan Lee)**
임상약학 석사 (아주대학교) | AI 개발자 과정 수료 (국비, MBC아카데미 수원)
GitHub: [@leesihwan21](https://github.com/leesihwan21)
