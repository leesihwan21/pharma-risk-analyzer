# app/routes/rag.py
import os
import re
import pickle
import anthropic
from flask import Blueprint, jsonify, request, render_template, current_app
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.models import db, RagHistory

rag = Blueprint('rag', __name__)

# ── 하드코딩 경로/모델명 제거 → current_app.config 사용 ──────────

# 전역 지연 로딩 객체
embeddings  = None
vectordb    = None
bm25_index  = None
bm25_corpus = None

def load_resources():
    global embeddings, vectordb, bm25_index, bm25_corpus
    if vectordb is None:
        rag_db_path = current_app.config['RAG_DB_PATH']
        embeddings  = HuggingFaceEmbeddings(
            model_name=current_app.config['EMBEDDING_MODEL'])
        vectordb    = FAISS.load_local(
            rag_db_path, embeddings, allow_dangerous_deserialization=True)

    if bm25_index is None:
        bm25_corpus_path = os.path.join(
            current_app.config['RAG_DB_PATH'], 'bm25_corpus.pkl')
        if os.path.exists(bm25_corpus_path):
            with open(bm25_corpus_path, 'rb') as f:
                bm25_corpus = pickle.load(f)
            bm25_index = BM25Okapi([d['tokens'] for d in bm25_corpus])

    return vectordb, bm25_index, bm25_corpus


def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    vdb, bm25_idx, corpus = load_resources()
    rrf_k = 60

    dense_results = vdb.similarity_search_with_score(query, k=k * 3)
    dense_scores  = {}
    for rank, (doc, _dist) in enumerate(dense_results):
        doc_id = doc.metadata.get('doc_id', doc.page_content[:50])
        dense_scores[doc_id] = {
            "rrf":      1 / (rank + rrf_k),
            "doc":      doc,
            "metadata": doc.metadata
        }

    sparse_scores = {}
    if bm25_idx and corpus:
        tokens     = query.lower().split()
        bm25_raw   = bm25_idx.get_scores(tokens)
        ranked_idx = sorted(range(len(bm25_raw)), key=lambda i: bm25_raw[i], reverse=True)[:k * 3]
        for rank, idx in enumerate(ranked_idx):
            if bm25_raw[idx] == 0:
                continue
            item   = corpus[idx]
            doc_id = item['metadata'].get('doc_id', item['content'][:50])
            sparse_scores[doc_id] = {
                "rrf":      1 / (rank + rrf_k),
                "content":  item['content'],
                "metadata": item['metadata']
            }

    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    fused   = []
    for doc_id in all_ids:
        d_score = dense_scores.get(doc_id, {}).get('rrf', 0) * alpha
        s_score = sparse_scores.get(doc_id, {}).get('rrf', 0) * (1 - alpha)
        total   = d_score + s_score

        if doc_id in dense_scores:
            doc      = dense_scores[doc_id]['doc']
            content  = doc.page_content
            metadata = doc.metadata
        else:
            content  = sparse_scores[doc_id]['content']
            metadata = sparse_scores[doc_id]['metadata']

        fused.append({"score": total, "content": content, "metadata": metadata})

    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:k]


SYSTEM_PROMPT = """You are an expert pharmacovigilance AI assistant with deep knowledge in:
- Drug safety monitoring and adverse event analysis
- FDA FAERS database interpretation
- ICH E2E pharmacovigilance guidelines
- Drug-drug interactions and contraindications
- Signal detection methods (PRR, ROR, BCPNN)
- Clinical risk-benefit assessment
- Regulatory reporting requirements (21 CFR Part 314.81)

STRICT RESPONSE RULES:
1. Answer ONLY in Korean
2. Base your answer SOLELY on the provided reference documents
3. If information is insufficient, clearly state: "참조 문서에서 확인할 수 없습니다"
4. ALWAYS cite your sources using: [출처: {doc_id}]
5. Highlight critical safety warnings with ⚠️ symbol"""


def call_anthropic_with_cache(context_text: str, question: str) -> dict:
    api_key = current_app.config['ANTHROPIC_API_KEY']
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=current_app.config['CLAUDE_MODEL'],
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[참고 문서]\n{context_text}",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"[질문]\n{question}"
                    }
                ]
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )

    usage = {
        "input_tokens":          response.usage.input_tokens,
        "output_tokens":         response.usage.output_tokens,
        "cache_creation_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
        "cache_read_tokens":     getattr(response.usage, 'cache_read_input_tokens', 0),
    }
    return {"answer": response.content[0].text, "usage": usage}


@rag.route('/rag')
def rag_page():
    return render_template('rag.html')


@rag.route('/api/rag/query', methods=['POST'])
def rag_query():
    data     = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': '질문을 입력해주세요'}), 400

    rag_top_k = current_app.config['RAG_TOP_K']

    try:
        results = hybrid_search(question, k=rag_top_k, alpha=0.6)
    except Exception as e:
        return jsonify({'error': f'검색 오류: {str(e)}'}), 500

    if not results:
        return jsonify({'error': '관련 문서를 찾을 수 없습니다'}), 404

    context_parts = []
    citations     = []
    for item in results:
        meta    = item['metadata']
        doc_id  = meta.get('doc_id', 'unknown')
        drug    = meta.get('drug', 'unknown')
        source  = meta.get('source', 'PubMed')
        content = item['content']

        context_parts.append(
            f"[문서 ID: {doc_id} | 약물: {drug} | 출처: {source}]\n{content}"
        )
        citations.append({
            "doc_id":  doc_id,
            "drug":    drug,
            "source":  source,
            "snippet": content[:200],
            "score":   round(item['score'], 4)
        })

    context_text = "\n\n---\n\n".join(context_parts)

    try:
        result = call_anthropic_with_cache(context_text, question)
        answer = result['answer']
        usage  = result['usage']
    except ValueError as e:
        import requests as http_requests
        ollama_url  = current_app.config['OLLAMA_URL']
        ai_timeout  = current_app.config['AI_TIMEOUT']
        try:
            resp   = http_requests.post(
                ollama_url,
                json={'model': 'llama3.2',
                      'prompt': f"[참고]\n{context_text[:1500]}\n\n[질문]\n{question}\n\n한국어로 답변:",
                      'stream': False},
                timeout=ai_timeout)
            answer = resp.json().get('response', '응답 생성 실패')
            usage  = {}
        except Exception as e2:
            return jsonify({'error': f'API 오류: {str(e2)}'}), 500
    except Exception as e:
        return jsonify({'error': f'API 오류: {str(e)}'}), 500

    try:
        log = RagHistory(
            question=question,
            answer=answer,
            sources=citations[0]['doc_id'] if citations else ''
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({
        'question':      question,
        'answer':        answer,
        'citations':     citations,
        'token_usage':   usage,
        'search_method': 'hybrid_rrf'
    })


@rag.route('/api/rag/history')
def rag_history():
    logs = RagHistory.query.order_by(RagHistory.asked_at.desc()).limit(20).all()
    return jsonify({'history': [l.to_dict() for l in logs]})