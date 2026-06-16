# RAG Evaluation

평가 질문 수: 5개 | Judge LLM: llama3.1:8b

> RAGAS 라이브러리의 의존성 충돌로 동일한 평가 방법론(Faithfulness, Answer Relevancy, Context Precision)을 Claude API 직접 호출로 구현했습니다.

## 평균 점수

| Metric | Score |
|---|---|
| Faithfulness | 0.62 |
| Answer Relevancy | 0.82 |
| Context Precision | 0.34 |

## 질문별 상세

| Question | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| 아스피린의 주요 부작용은 무엇인가요? | 0.8 | 0.8 | 0.5 |
| 메트포르민 복용 시 주의해야 할 점은? | 0.8 | 0.8 | 0.5 |
| 이부프로펜과 관련된 심혈관계 위험이 있나요? | 0.5 | 1.0 | 0.0 |
| 와파린 복용 중 출혈 위험은 어떻게 관리하나요? | 0.2 | 0.5 | 0.5 |
| 스타틴 계열 약물의 간 독성 가능성에 대해 알려주세요. | 0.8 | 1.0 | 0.2 |
