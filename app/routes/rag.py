import os
import requests as http_requests
from flask import Blueprint, jsonify, request, render_template
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

rag = Blueprint('rag', __name__)

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
        db = load_vectordb()
        docs = db.similarity_search(question, k=3)
        context = '\n\n'.join([doc.page_content for doc in docs])
    except Exception as e:
        return jsonify({'error': f'벡터DB 로드 실패: {str(e)}'}), 500

    prompt = f"""반드시 한국어로만 답하세요. 아래 논문에 없는 내용은 절대 지어내지 마세요. 모르면 "논문에서 확인되지 않습니다"라고 답하세요.

[참고 논문]
{context[:2000]}

[질문]
{question}

[답변] 논문 내용만 근거로 3문장 이내로 한국어 답변:"""

    try:
        response = http_requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt, 'stream': False},
            timeout=120
        )
        answer = response.json().get('response', '답변 생성 실패')
    except Exception as e:
        answer = f'Ollama 오류: {str(e)}'

    return jsonify({
        'question': question,
        'answer': answer,
        'sources': [doc.page_content[:200] for doc in docs]
    })