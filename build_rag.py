import requests
import pickle
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 검색할 약물 목록
DRUGS = [
    'methotrexate', 'cyclophosphamide', 'doxorubicin', 'paclitaxel', 'carboplatin',
    'adalimumab', 'infliximab', 'rituximab', 'etanercept', 'tocilizumab',
    'warfarin', 'aspirin', 'clopidogrel', 'atorvastatin', 'lisinopril',
    'ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'indomethacin',
    'amoxicillin', 'ciprofloxacin', 'azithromycin', 'vancomycin', 'metronidazole',
    'omeprazole', 'metformin', 'levothyroxine', 'prednisone', 'gabapentin'
]

def fetch_pubmed(drug, max_results=5):
    print(f"  PubMed 검색 중: {drug}")
    search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {'db': 'pubmed', 'term': f'{drug} adverse event safety', 'retmax': max_results, 'retmode': 'json'}
    ids = requests.get(search_url, params=params, timeout=10).json()['esearchresult']['idlist']
    if not ids:
        return ''
    fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {'db': 'pubmed', 'id': ','.join(ids), 'rettype': 'abstract', 'retmode': 'text'}
    return requests.get(fetch_url, params=params, timeout=10).text

# 논문 수집
print("=== PubMed 논문 수집 ===")
all_texts = []
for drug in DRUGS:
    text = fetch_pubmed(drug)
    if text:
        all_texts.append(f"[{drug.upper()}]\n{text}")
        print(f"  {drug}: {len(text)}자 수집")

# 텍스트 청크 분할
print("\n=== 텍스트 청크 분할 ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(all_texts)
print(f"  총 {len(chunks)}개 청크 생성")

# 임베딩 + FAISS 벡터DB 생성
print("\n=== FAISS 벡터DB 생성 (시간 좀 걸려) ===")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb = FAISS.from_documents(chunks, embeddings)

# 저장
os.makedirs('rag_db', exist_ok=True)
vectordb.save_local('rag_db')
print("\n✅ 완료! rag_db/ 폴더에 저장됨")

# 테스트 검색
print("\n=== 검색 테스트 ===")
query = "methotrexate 부작용 위험"
docs = vectordb.similarity_search(query, k=3)
for i, doc in enumerate(docs):
    print(f"\n[{i+1}] {doc.page_content[:200]}")