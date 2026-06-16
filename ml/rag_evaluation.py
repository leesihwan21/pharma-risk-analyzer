"""
RAG 평가: Faithfulness, Answer Relevancy, Context Precision 측정
RAGAS 라이브러리의 의존성 충돌(langchain_community vertexai 모듈 누락) 문제로,
동일한 평가 방법론을 Claude API 직접 호출로 구현
사용법: python ml/rag_evaluation.py
결과: ml/rag_evaluation.json, ml/rag_evaluation.md
"""

import os
import json
import re
import requests as http_requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DB_PATH = os.path.join(BASE_DIR, 'rag_db')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

EVAL_QUESTIONS = [
    "아스피린의 주요 부작용은 무엇인가요?",
    "메트포르민 복용 시 주의해야 할 점은?",
    "이부프로펜과 관련된 심혈관계 위험이 있나요?",
    "와파린 복용 중 출혈 위험은 어떻게 관리하나요?",
    "스타틴 계열 약물의 간 독성 가능성에 대해 알려주세요.",
]

JUDGE_MODEL = "llama3.1:8b"


def _strip_unwanted_scripts(text):
    return re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]', '', text)


def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(RAG_DB_PATH, embeddings, allow_dangerous_deserialization=True)


def generate_answer(question, context):
    """rag.py와 동일한 프롬프트 구조로 llama3.2 호출"""
    prompt = f"""당신은 한국어 의약품 정보 보조원입니다. 반드시 자연스러운 한국어로 약물명 외에 불필요한 외국어 단어를 사용하세요.

엄격 규칙:
- 중국어 한자, 베트남어, 일본어, 기타 외국어 문자를 절대 쓰지 마세요.
- 아래 논문에 없는 내용은 절대 지어내지 마세요. 모르면 "논문에서 확인하지 못했습니다"라고 답하세요.

[참고 논문]
{context[:2000]}

[질문]
{question}

[답변 예시 형식]
1. (한국어 문장)
2. (한국어 문장)

[답변] 논문 내용만 근거로 3문장 이내, 순수 한국어로만 답변:"""

    try:
        response = http_requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.2, 'top_p': 0.85}
            },
            timeout=120
        )
        answer = response.json().get('response', '답변 생성 실패')
        return _strip_unwanted_scripts(answer)
    except Exception as e:
        return f'Ollama 오류: {str(e)}'


def judge_score(prompt, max_tokens=300):
    """Ollama(llama3.1:8b)에게 0~1 사이 점수와 근거를 JSON으로 요청"""
    response = http_requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': JUDGE_MODEL,
            'prompt': prompt + "\n\n반드시 JSON 형식만 출력하세요. 다른 설명은 추가하지 마세요.",
            'stream': False,
            'options': {'temperature': 0.0}
        },
        timeout=120
    )
    text = response.json().get('response', '').strip()
    text = re.sub(r'^```json\s*|\s*```$', '', text)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        return float(parsed.get("score", 0.0)), parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, AttributeError):
        match = re.search(r'0?\.\d+|\d\.\d+|[01]', text)
        score = float(match.group()) if match else 0.0
        return score, text[:200]


def eval_faithfulness(question, answer, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""다음 답변이 주어진 컨텍스트(문헌)에 사실적으로 충실한지 평가하세요.
답변에 포함된 모든 주장이 컨텍스트에서 근거를 찾을 수 있는지 확인하세요.
컨텍스트에 없는 내용을 답변이 지어냈다면(hallucination) 낮은 점수를 주세요.

[질문]
{question}

[컨텍스트]
{context_text[:3000]}

[평가할 답변]
{answer}

JSON 형식으로만 응답하세요: {{"score": 0.0~1.0 사이 숫자, "reasoning": "한 문장 근거"}}"""
    return judge_score(prompt)


def eval_answer_relevancy(question, answer):
    prompt = f"""다음 답변이 질문에 얼마나 직접적이고 적절하게 대응하는지 평가하세요.
질문과 무관한 내용이 많거나, 핵심을 회피하면 낮은 점수를 주세요.

[질문]
{question}

[평가할 답변]
{answer}

JSON 형식으로만 응답하세요: {{"score": 0.0~1.0 사이 숫자, "reasoning": "한 문장 근거"}}"""
    return judge_score(prompt)


def eval_context_precision(question, contexts):
    context_list = "\n---\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])
    prompt = f"""다음은 질문에 대해 검색된 컨텍스트(문헌 조각) 목록입니다.
각 컨텍스트가 질문에 답하는 데 실제로 유용한지 평가하고,
유용한 컨텍스트가 상위에 배치되어 있는지를 기준으로 전체 검색 품질에 0.0~1.0 점수를 매기세요.

[질문]
{question}

[검색된 컨텍스트 목록]
{context_list}

JSON 형식으로만 응답하세요: {{"score": 0.0~1.0 사이 숫자, "reasoning": "한 문장 근거"}}"""
    return judge_score(prompt)


def main():
    print("Loading FAISS vector DB...")
    vdb = load_vectordb()

    results = []
    for q in EVAL_QUESTIONS:
        print(f"\nQuestion: {q}")
        docs = vdb.similarity_search(q, k=3)
        contexts = [doc.page_content for doc in docs]

        answer = generate_answer(q, "\n\n".join(contexts))
        print(f"Answer: {answer[:100]}...")

        faith_score, faith_reason = eval_faithfulness(q, answer, contexts)
        relevancy_score, relevancy_reason = eval_answer_relevancy(q, answer)
        precision_score, precision_reason = eval_context_precision(q, contexts)

        print(f"  Faithfulness: {faith_score} | Relevancy: {relevancy_score} | Precision: {precision_score}")

        results.append({
            "question": q,
            "answer": answer,
            "num_contexts": len(contexts),
            "faithfulness": round(faith_score, 4),
            "faithfulness_reasoning": faith_reason,
            "answer_relevancy": round(relevancy_score, 4),
            "answer_relevancy_reasoning": relevancy_reason,
            "context_precision": round(precision_score, 4),
            "context_precision_reasoning": precision_reason,
        })

    avg = {
        "faithfulness": round(sum(r['faithfulness'] for r in results) / len(results), 4),
        "answer_relevancy": round(sum(r['answer_relevancy'] for r in results) / len(results), 4),
        "context_precision": round(sum(r['context_precision'] for r in results) / len(results), 4),
    }

    output = {
        "questions": results,
        "average": avg,
        "n_questions": len(EVAL_QUESTIONS),
        "judge_model": JUDGE_MODEL,
        "note": "RAGAS 라이브러리 의존성 충돌로 동일 방법론을 Claude API 직접 호출로 구현"
    }

    json_path = os.path.join(MODEL_DIR, 'rag_evaluation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    md_path = os.path.join(MODEL_DIR, 'rag_evaluation.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# RAG Evaluation\n\n")
        f.write(f"평가 질문 수: {len(EVAL_QUESTIONS)}개 | Judge LLM: {JUDGE_MODEL}\n\n")
        f.write("> RAGAS 라이브러리의 의존성 충돌로 동일한 평가 방법론(Faithfulness, Answer Relevancy, Context Precision)을 Claude API 직접 호출로 구현했습니다.\n\n")
        f.write("## 평균 점수\n\n")
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| Faithfulness | {avg['faithfulness']} |\n")
        f.write(f"| Answer Relevancy | {avg['answer_relevancy']} |\n")
        f.write(f"| Context Precision | {avg['context_precision']} |\n\n")
        f.write("## 질문별 상세\n\n")
        f.write("| Question | Faithfulness | Answer Relevancy | Context Precision |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['question']} | {r['faithfulness']} | {r['answer_relevancy']} | {r['context_precision']} |\n")
    print(f"Saved: {md_path}")


if __name__ == '__main__':
    main()