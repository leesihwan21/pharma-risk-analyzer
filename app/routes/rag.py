import os
import re
import requests as http_requests
from flask import Blueprint, jsonify, request, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.models import db, RagHistory

rag = Blueprint('rag', __name__)

def _strip_unwanted_scripts(text):
    """한자(CJK 통합 한자), 힌디어(데바나가리) 등 한국어 답변에 부적절한 문자 제거"""
    return re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]', '', text)


RAG_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'rag_db')

embeddings = None
vectordb = None

def load_vectordb():
    global embeddings, vectordb
    if vectordb is None:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectordb = FAISS.load_local(RAG_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectordb

@rag.route('/rag')
def rag_page():
    return render_template('rag.html')

@rag.route('/api/rag/query', methods=['POST'])
def rag_query():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': '질문을 입력하세요'}), 400

    try:
        vdb = load_vectordb()
        docs = vdb.similarity_search(question, k=3)
        context = '\n\n'.join([doc.page_content for doc in docs])
    except Exception as e:
        return jsonify({'error': f'벡터DB 로드 실패: {str(e)}'}), 500

    prompt = f"""당신은 한국어 의약 정보 보조원입니다. 반드시 자연스러운 한국어와 약물명 등 필요한 영어 단어만 사용하세요.
절대 규칙:
- 중국어 한자, 힌디어, 일본어, 기타 외국어 문자를 절대 섞지 마세요.
- 아래 논문에 없는 내용은 절대 지어내지 마세요. 모르면 "논문에서 확인되지 않습니다"라고 답하세요.

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
        answer = _strip_unwanted_scripts(answer)
    except Exception as e:
        answer = f'Ollama 오류: {str(e)}'

    try:
        log = RagHistory(
            question=question,
            answer=answer,
            sources=docs[0].page_content[:200] if docs else ''
        )
        db.session.add(log)
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({
        'question': question,
        'answer': answer,
        'sources': [doc.page_content[:200] for doc in docs]
    })

@rag.route('/api/rag/history')
def rag_history():
    logs = RagHistory.query.order_by(RagHistory.asked_at.desc()).limit(20).all()
    return jsonify({'history': [l.to_dict() for l in logs]})