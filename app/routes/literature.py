import os
import requests
import xml.etree.ElementTree as ET
from flask import Blueprint, render_template, jsonify, request
from app import cache

literature = Blueprint('literature', __name__)

def search_pubmed(query, max_results=5):
    """PubMed에서 논문 ID 검색"""
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json'
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    ids = data['esearchresult']['idlist']
    total = data['esearchresult']['count']
    return ids, total

def fetch_pubmed_details(pmids):
    """PubMed에서 논문 상세 정보 가져오기"""
    if not pmids:
        return []
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
        'rettype': 'abstract'
    }
    r = requests.get(url, params=params, timeout=15)
    root = ET.fromstring(r.text)
    papers = []
    for article in root.findall('.//PubmedArticle'):
        try:
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle').text or 'No title'
            # 초록 추출
            abstract_parts = article.findall('.//AbstractText')
            abstract = ' '.join([p.text for p in abstract_parts if p.text]) if abstract_parts else 'No abstract available'
            # 저널명
            journal = article.find('.//Title')
            journal_name = journal.text if journal is not None else 'Unknown Journal'
            # 발행연도
            year = article.find('.//PubDate/Year')
            pub_year = year.text if year is not None else 'N/A'
            # DOI
            doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else None
            papers.append({
                'pmid': pmid,
                'title': title,
                'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                'full_abstract': abstract,
                'journal': journal_name,
                'year': pub_year,
                'doi': doi,
                'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
            })
        except Exception as e:
            continue
    return papers

def summarize_with_claude(papers, drug_name):
    """Claude API로 논문 요약"""
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return 'API key not configured.'
    abstracts = '\n\n'.join([
        f"[{i+1}] {p['title']} ({p['year']})\n{p['full_abstract']}"
        for i, p in enumerate(papers[:3])
    ])
    prompt = f"""You are a clinical pharmacologist. Based on the following PubMed abstracts about {drug_name}, provide a concise summary in Korean (3-4 sentences) covering:
1. Main adverse events reported
2. Key risk factors
3. Clinical implications

Abstracts:
{abstracts}

Summary (in Korean):"""

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=500,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return message.content[0].text

@literature.route('/literature')
def literature_page():
    return render_template('literature.html')

@literature.route('/api/literature/search')
def search_literature():
    drug = request.args.get('drug', '').strip()
    if not drug:
        return jsonify({'error': '약물명을 입력해주세요.'}), 400

    try:
        query = f'{drug} adverse event pharmacovigilance'
        pmids, total = search_pubmed(query, max_results=5)
        papers = fetch_pubmed_details(pmids)

        summary = ''
        if papers:
            try:
                summary = summarize_with_claude(papers, drug)
            except Exception as e:
                summary = f'요약 생성 실패: {str(e)}'

        return jsonify({
            'drug': drug,
            'total_found': total,
            'papers': papers,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
