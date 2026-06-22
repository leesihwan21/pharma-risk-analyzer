# build_rag.py
import requests
import pickle
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DRUGS = [
    'methotrexate', 'cyclophosphamide', 'doxorubicin', 'paclitaxel', 'carboplatin',
    'adalimumab', 'infliximab', 'rituximab', 'etanercept', 'tocilizumab',
    'warfarin', 'aspirin', 'clopidogrel', 'atorvastatin', 'lisinopril',
    'ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'indomethacin',
    'amoxicillin', 'ciprofloxacin', 'azithromycin', 'vancomycin', 'metronidazole',
    'omeprazole', 'metformin', 'levothyroxine', 'prednisone', 'gabapentin'
]

def fetch_pubmed(drug, max_results=5):
    print(f"  PubMed 寃??以? {drug}")
    try:
        search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
        params = {'db': 'pubmed', 'term': f'{drug} adverse event safety', 'retmax': max_results, 'retmode': 'json'}
        ids = requests.get(search_url, params=params, timeout=10).json()['esearchresult']['idlist']
        if not ids:
            return ''
        fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        params = {'db': 'pubmed', 'id': ','.join(ids), 'rettype': 'abstract', 'retmode': 'text'}
        return requests.get(fetch_url, params=params, timeout=10).text
    except Exception as e:
        print(f"  [{drug}] ?ㅻ쪟: {e}")
        return ''

print("=== PubMed ?곗씠???섏쭛 ===")
raw_texts = []  # (drug, text) ?쒗뵆 由ъ뒪??
for drug in DRUGS:
    text = fetch_pubmed(drug)
    if text:
        raw_texts.append((drug, text))
        print(f"  {drug}: {len(text)}???섏쭛")

# 泥?겕 遺꾪븷 + 硫뷀??곗씠??遺李?
print("\n=== 泥?겕 遺꾪븷 + 硫뷀??곗씠??遺李?===")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = []
for drug, text in raw_texts:
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "drug":      drug,
                "source":    "PubMed",
                "doc_id":    f"pubmed_{drug}_{i}",
                "chunk_idx": i
            }
        )
        documents.append(doc)

print(f"  珥?{len(documents)}媛?泥?겕 ?앹꽦")

# FAISS 踰≫꽣DB ???
print("\n=== FAISS 踰≫꽣DB ?앹꽦 (?쒓컙 ?뚯슂) ===")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb = FAISS.from_documents(documents, embeddings)
os.makedirs('rag_db', exist_ok=True)
vectordb.save_local('rag_db')
print("  FAISS ????꾨즺")

# BM25 肄뷀띁?????(?좏겙 由ъ뒪??+ 硫뷀??곗씠??
print("\n=== BM25 肄뷀띁?????===")
bm25_corpus = []
for doc in documents:
    bm25_corpus.append({
        "tokens":   doc.page_content.lower().split(),
        "content":  doc.page_content,
        "metadata": doc.metadata
    })

with open('rag_db/bm25_corpus.pkl', 'wb') as f:
    pickle.dump(bm25_corpus, f)
print(f"  BM25 肄뷀띁??{len(bm25_corpus)}媛?????꾨즺")

# 寃???뚯뒪??
print("\n=== 寃???뚯뒪??===")
query = "methotrexate adverse event hepatotoxicity"
docs = vectordb.similarity_search(query, k=3)
for i, doc in enumerate(docs):
    print(f"\n[{i+1}] ?쎈Ъ: {doc.metadata['drug']} | 異쒖쿂: {doc.metadata['source']} | ID: {doc.metadata['doc_id']}")
    print(f"     {doc.page_content[:150]}...")

print("\n=== ?꾨즺! rag_db/ ?대뜑????λ맖 ===")
