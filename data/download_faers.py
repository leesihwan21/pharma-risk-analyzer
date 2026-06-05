import requests
import zipfile
import os

# 저장 경로
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# ✅ 멀티쿼터 설정 — 쿼터 추가/제거는 여기서만 하면 됨
QUARTERS = [
    '2024q1',
    '2024q2',
    '2024q3',
    '2025q1',
]

def get_faers_url(quarter: str) -> str:
    """쿼터명으로 FDA FAERS 다운로드 URL 생성"""
    return f'https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip'

def download_quarter(quarter: str):
    """단일 쿼터 다운로드 및 압축 해제"""
    name = f'faers_ascii_{quarter}'
    zip_path = os.path.join(RAW_DIR, f'{name}.zip')
    extract_path = os.path.join(RAW_DIR, name)

    # 이미 압축 해제된 폴더가 있으면 스킵
    if os.path.exists(extract_path):
        print(f'⏭️  이미 존재함, 스킵: {name}')
        return

    url = get_faers_url(quarter)
    print(f'📥 다운로드 중: {name}')

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f'❌ 다운로드 실패 (HTTP {response.status_code}): {name}')
        return

    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            percent = (downloaded / total * 100) if total else 0
            print(f'\r  진행률: {percent:.1f}%', end='', flush=True)

    print(f'\n✅ 다운로드 완료: {zip_path}')

    # 압축 해제
    print(f'📦 압축 해제 중...')
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f'✅ 압축 해제 완료: {extract_path}')

def download_faers():
    """QUARTERS 리스트의 모든 쿼터 다운로드"""
    print(f'🚀 총 {len(QUARTERS)}개 쿼터 다운로드 시작\n')
    for quarter in QUARTERS:
        download_quarter(quarter)
        print()
    print('🎉 전체 다운로드 완료!')

if __name__ == '__main__':
    download_faers()
