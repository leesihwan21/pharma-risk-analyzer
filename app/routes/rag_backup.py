# app/routes/rag.py
import os
import pickle
import anthropic
from flask import Blueprint, jsonify, request, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.models import db, RagHistory

rag = Blueprint("rag", __name__)

# ── 경로 설정 ──────────────────────────────────────────────
RAG_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rag_db",
)
BM25_CORPUS_PATH = os.path.join(RAG_DB_PATH, "bm25_corpus.pkl")

# ── 전역 객체 (앱 시작 시 1회 로드) ────────────────────────
embeddings = None
vectordb = None
bm25_index = None
bm25_corpus = None


def load_resources():
    global embeddings, vectordb, bm25_index, bm25_corpus
    if vectordb is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.load_local(
            RAG_DB_PATH, embeddings, allow_dangerous_deserialization=True
        )
    if bm25_index is None and os.path.exists(BM25_CORPUS_PATH):
        with open(BM25_CORPUS_PATH, "rb") as f:
            bm25_corpus = pickle.load(f)
        bm25_index = BM25Okapi([d["tokens"] for d in bm25_corpus])
    return vectordb, bm25_index, bm25_corpus


# ── 하이브리드 검색 (RRF: Reciprocal Rank Fusion) ─────────
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    """
    alpha: FAISS 가중치 (1-alpha: BM25 가중치)
    RRF 공식: score = 1 / (rank + 60)
    """
    vdb, bm25_idx, corpus = load_resources()
    rrf_k = 60  # RRF 상수

    # ── Dense 검색 (FAISS) ──
    dense_results = vdb.similarity_search_with_score(query, k=k * 3)
    dense_scores = {}
    for rank, (doc, _dist) in enumerate(dense_results):
        doc_id = doc.metadata.get("doc_id", doc.page_content[:50])
        dense_scores[doc_id] = {
            "rrf": 1 / (rank + rrf_k),
            "doc": doc,
            "metadata": doc.metadata,
        }

    # ── Sparse 검색 (BM25) ──
    sparse_scores = {}
    if bm25_idx and corpus:
        tokens = query.lower().split()
        bm25_raw = bm25_idx.get_scores(tokens)
        ranked_idx = sorted(
            range(len(bm25_raw)), key=lambda i: bm25_raw[i], reverse=True
        )[: k * 3]
        for rank, idx in enumerate(ranked_idx):
            if bm25_raw[idx] == 0:
                continue
            item = corpus[idx]
            doc_id = item["metadata"].get("doc_id", item["content"][:50])
            sparse_scores[doc_id] = {
                "rrf": 1 / (rank + rrf_k),
                "content": item["content"],
                "metadata": item["metadata"],
            }

    # ── RRF 융합 ──
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    fused = []
    for doc_id in all_ids:
        d_score = dense_scores.get(doc_id, {}).get("rrf", 0) * alpha
        s_score = sparse_scores.get(doc_id, {}).get("rrf", 0) * (1 - alpha)
        total = d_score + s_score

        if doc_id in dense_scores:
            doc = dense_scores[doc_id]["doc"]
            content = doc.page_content
            metadata = doc.metadata
        else:
            content = sparse_scores[doc_id]["content"]
            metadata = sparse_scores[doc_id]["metadata"]

        fused.append({"score": total, "content": content, "metadata": metadata})

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]


# ── 시스템 프롬프트 (캐싱 대상 - 고정 텍스트) ──────────────
SYSTEM_PROMPT = """You are an expert pharmacovigilance AI assistant with deep knowledge in:
- Drug safety monitoring and adverse event analysis
- FDA FAERS database interpretation
- ICH E2E pharmacovigilance guidelines
- Drug-drug interactions and contraindications
- Signal detection methods (PRR, ROR, BCPNN)
- Clinical risk-benefit assessment
- Regulatory reporting requirements (21 CFR Part 314.81)

Your role is to analyze drug safety information and provide evidence-based responses.

STRICT RESPONSE RULES:
1. Answer ONLY in Korean (한국어로만 답변)
2. Do NOT use Chinese characters (한자), Devanagari, or any non-Korean/English scripts
3. Base your answer SOLELY on the provided reference documents
4. If information is insufficient, clearly state: "제공된 문서에서 확인할 수 없습니다"
5. ALWAYS cite your sources using the exact format: [출처: {doc_id}]
6. Structure your response as numbered points when listing multiple items
7. Highlight critical safety warnings with ⚠️ symbol
8. Keep medical terminology accurate; use Korean terms with English in parentheses

CITATION FORMAT (mandatory):
- Every factual claim must end with [출처: {doc_id}]
- Example: "메토트렉세이트는 간독성을 유발할 수 있습니다 [출처: pubmed_methotrexate_2]"

RESPONSE STRUCTURE:
1. 핵심 답변 (Key Answer)
2. 상세 설명 (Details) - with citations
3. ⚠️ 안전 주의사항 (Safety Warnings) if applicable"""


# ── Anthropic API 호출 (프롬프트 캐싱 적용) ───────────────
def call_anthropic_with_cache(context_text: str, question: str) -> dict:
    """
    캐싱 전략:
    - system prompt: ephemeral 캐시 (고정 텍스트, 반복 질의 시 토큰 절약)
    - context: ephemeral 캐시 (같은 검색 결과 반복 시 절약)
    - question: 캐시 없음 (매번 변동)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # ← 시스템 프롬프트 캐싱
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[참고 문서]\n{context_text}",
                        "cache_control": {"type": "ephemeral"},  # ← 컨텍스트 캐싱
                    },
                    {"type": "text", "text": f"[질문]\n{question}"},  # ← 캐시 없음
                ],
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    answer = response.content[0].text

    # 토큰 사용량 추출 (캐시 히트 여부 포함)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0
        ),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }
    return {"answer": answer, "usage": usage}


# ── 라우트 ─────────────────────────────────────────────────
@rag.route("/rag")
def rag_page():
    return render_template("rag.html")


@rag.route("/api/rag/query", methods=["POST"])
def rag_query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400

    # ── 하이브리드 검색 ──
    try:
        results = hybrid_search(question, k=5, alpha=0.6)
    except Exception as e:
        return jsonify({"error": f"검색 오류: {str(e)}"}), 500

    if not results:
        return jsonify({"error": "관련 문서를 찾을 수 없습니다"}), 404

    # ── 컨텍스트 + 출처 정보 구성 ──
    context_parts = []
    citations = []
    for item in results:
        meta = item["metadata"]
        doc_id = meta.get("doc_id", "unknown")
        drug = meta.get("drug", "unknown")
        source = meta.get("source", "PubMed")
        content = item["content"]

        context_parts.append(
            f"[문서 ID: {doc_id} | 약물: {drug} | 출처: {source}]\n{content}"
        )
        citations.append(
            {
                "doc_id": doc_id,
                "drug": drug,
                "source": source,
                "snippet": content[:200],
                "score": round(item["score"], 4),
            }
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # ── Anthropic API 호출 ──
    try:
        result = call_anthropic_with_cache(context_text, question)
        answer = result["answer"]
        usage = result["usage"]
    except ValueError:
        # API 키 없으면 Ollama 폴백
        import requests as http_requests

        try:
            resp = http_requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": f"[참고]\n{context_text[:1500]}\n\n[질문]\n{question}\n\n한국어로 답변:",
                    "stream": False,
                },
                timeout=120,
            )
            answer = resp.json().get("response", "응답 생성 실패")
            usage = {}
        except Exception as e2:
            return jsonify({"error": f"API 오류: {str(e2)}"}), 500
    except Exception as e:
        return jsonify({"error": f"API 오류: {str(e)}"}), 500

    # ── DB 로깅 ──
    try:
        log = RagHistory(
            question=question,
            answer=answer,
            sources=citations[0]["doc_id"] if citations else "",
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "citations": citations,  # 출처 목록
            "token_usage": usage,  # 캐시 히트 여부 포함
            "search_method": "hybrid_rrf",  # 검색 방법 명시
        }
    )


@rag.route("/api/rag/history")
def rag_history():
    logs = RagHistory.query.order_by(RagHistory.asked_at.desc()).limit(20).all()
    return jsonify({"history": [l.to_dict() for l in logs]})
