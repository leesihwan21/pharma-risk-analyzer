# app/routes/rag.py
import os
import re
import pickle
import anthropic
from flask import Blueprint, jsonify, request, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.models import db, RagHistory

rag = Blueprint('rag', __name__)

# ?? 寃쎈줈 ?ㅼ젙 ??????????????????????????????????????????????
RAG_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'rag_db'
)
BM25_CORPUS_PATH = os.path.join(RAG_DB_PATH, 'bm25_corpus.pkl')

# ?? ?꾩뿭 媛앹껜 (???쒖옉 ??1??濡쒕뱶) ????????????????????????
embeddings  = None
vectordb    = None
bm25_index  = None
bm25_corpus = None

def load_resources():
    global embeddings, vectordb, bm25_index, bm25_corpus
    if vectordb is None:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectordb   = FAISS.load_local(RAG_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    if bm25_index is None and os.path.exists(BM25_CORPUS_PATH):
        with open(BM25_CORPUS_PATH, 'rb') as f:
            bm25_corpus = pickle.load(f)
        bm25_index = BM25Okapi([d['tokens'] for d in bm25_corpus])
    return vectordb, bm25_index, bm25_corpus


# ?? ?섏씠釉뚮━??寃??(RRF: Reciprocal Rank Fusion) ?????????
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    """
    alpha: FAISS 媛以묒튂 (1-alpha: BM25 媛以묒튂)
    RRF 怨듭떇: score = 1 / (rank + 60)
    """
    vdb, bm25_idx, corpus = load_resources()
    rrf_k = 60  # RRF ?곸닔

    # ?? Dense 寃??(FAISS) ??
    dense_results = vdb.similarity_search_with_score(query, k=k * 3)
    dense_scores  = {}
    for rank, (doc, _dist) in enumerate(dense_results):
        doc_id = doc.metadata.get('doc_id', doc.page_content[:50])
        dense_scores[doc_id] = {
            "rrf":      1 / (rank + rrf_k),
            "doc":      doc,
            "metadata": doc.metadata
        }

    # ?? Sparse 寃??(BM25) ??
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

    # ?? RRF ?듯빀 ??
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

        fused.append({
            "score":    total,
            "content":  content,
            "metadata": metadata
        })

    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:k]


# ?? ?쒖뒪???꾨＼?꾪듃 (罹먯떛 ???- 怨좎젙 ?띿뒪?? ??????????????
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
1. Answer ONLY in Korean (?쒓뎅?대줈留??듬?)
2. Do NOT use Chinese characters (?쒖옄), Devanagari, or any non-Korean/English scripts
3. Base your answer SOLELY on the provided reference documents
4. If information is insufficient, clearly state: "?쒓났??臾몄꽌?먯꽌 ?뺤씤?????놁뒿?덈떎"
5. ALWAYS cite your sources using the exact format: [異쒖쿂: {doc_id}]
6. Structure your response as numbered points when listing multiple items
7. Highlight critical safety warnings with ?좑툘 symbol
8. Keep medical terminology accurate; use Korean terms with English in parentheses

CITATION FORMAT (mandatory):
- Every factual claim must end with [異쒖쿂: {doc_id}]
- Example: "硫뷀넗?몃젆?몄씠?몃뒗 媛꾨룆?깆쓣 ?좊컻?????덉뒿?덈떎 [異쒖쿂: pubmed_methotrexate_2]"

RESPONSE STRUCTURE:
1. ?듭떖 ?듬? (Key Answer)
2. ?곸꽭 ?ㅻ챸 (Details) - with citations
3. ?좑툘 ?덉쟾 二쇱쓽?ы빆 (Safety Warnings) if applicable"""


# ?? Anthropic API ?몄텧 (?꾨＼?꾪듃 罹먯떛 ?곸슜) ???????????????
def call_anthropic_with_cache(context_text: str, question: str) -> dict:
    """
    罹먯떛 ?꾨왂:
    - system prompt: ephemeral 罹먯떆 (怨좎젙 ?띿뒪?? 諛섎났 吏덉쓽 ???좏겙 ?덉빟)
    - context: ephemeral 罹먯떆 (媛숈? 寃??寃곌낵 諛섎났 ???덉빟)
    - question: 罹먯떆 ?놁쓬 (留ㅻ쾲 蹂??
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}   # ???쒖뒪???꾨＼?꾪듃 罹먯떛
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[李멸퀬 臾몄꽌]\n{context_text}",
                        "cache_control": {"type": "ephemeral"}  # ??而⑦뀓?ㅽ듃 罹먯떛
                    },
                    {
                        "type": "text",
                        "text": f"[吏덈Ц]\n{question}"           # ??罹먯떆 ?놁쓬
                    }
                ]
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )

    answer = response.content[0].text

    # ?좏겙 ?ъ슜??異붿텧 (罹먯떆 ?덊듃 ?щ? ?ы븿)
    usage = {
        "input_tokens":         response.usage.input_tokens,
        "output_tokens":        response.usage.output_tokens,
        "cache_creation_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
        "cache_read_tokens":     getattr(response.usage, 'cache_read_input_tokens', 0),
    }
    return {"answer": answer, "usage": usage}


# ?? ?쇱슦???????????????????????????????????????????????????
@rag.route('/rag')
def rag_page():
    return render_template('rag.html')


@rag.route('/api/rag/query', methods=['POST'])
def rag_query():
    data     = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': '吏덈Ц???낅젰?댁＜?몄슂'}), 400

    # ?? ?섏씠釉뚮━??寃????
    try:
        results = hybrid_search(question, k=5, alpha=0.6)
    except Exception as e:
        return jsonify({'error': f'寃???ㅻ쪟: {str(e)}'}), 500

    if not results:
        return jsonify({'error': '愿??臾몄꽌瑜?李얠쓣 ???놁뒿?덈떎'}), 404

    # ?? 而⑦뀓?ㅽ듃 + 異쒖쿂 ?뺣낫 援ъ꽦 ??
    context_parts = []
    citations     = []
    for item in results:
        meta    = item['metadata']
        doc_id  = meta.get('doc_id', 'unknown')
        drug    = meta.get('drug', 'unknown')
        source  = meta.get('source', 'PubMed')
        content = item['content']

        context_parts.append(
            f"[臾몄꽌 ID: {doc_id} | ?쎈Ъ: {drug} | 異쒖쿂: {source}]\n{content}"
        )
        citations.append({
            "doc_id":  doc_id,
            "drug":    drug,
            "source":  source,
            "snippet": content[:200],
            "score":   round(item['score'], 4)
        })

    context_text = "\n\n---\n\n".join(context_parts)

    # ?? Anthropic API ?몄텧 ??
    try:
        result  = call_anthropic_with_cache(context_text, question)
        answer  = result['answer']
        usage   = result['usage']
    except ValueError as e:
        # API ???놁쑝硫?Ollama ?대갚
        import requests as http_requests
        try:
            resp   = http_requests.post(
                'http://localhost:11434/api/generate',
                json={'model': 'llama3.2', 'prompt': f"[李멸퀬]\n{context_text[:1500]}\n\n[吏덈Ц]\n{question}\n\n?쒓뎅?대줈 ?듬?:", 'stream': False},
                timeout=120
            )
            answer = resp.json().get('response', '?묐떟 ?앹꽦 ?ㅽ뙣')
            usage  = {}
        except Exception as e2:
            return jsonify({'error': f'API ?ㅻ쪟: {str(e2)}'}), 500
    except Exception as e:
        return jsonify({'error': f'API ?ㅻ쪟: {str(e)}'}), 500

    # ?? DB 濡쒓퉭 ??
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
        'question':  question,
        'answer':    answer,
        'citations': citations,          # 異쒖쿂 紐⑸줉
        'token_usage': usage,            # 罹먯떆 ?덊듃 ?щ? ?ы븿
        'search_method': 'hybrid_rrf'    # 寃??諛⑸쾿 紐낆떆
    })


@rag.route('/api/rag/history')
def rag_history():
    logs = RagHistory.query.order_by(RagHistory.asked_at.desc()).limit(20).all()
    return jsonify({'history': [l.to_dict() for l in logs]})
