import requests
import zipfile
import os

# 저장 경로
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# FDA FAERS 최신 분기 데이터 URL (2024 Q3)
# FDA FAERS 최신 분기 데이터 URL (2024 Q3)
FAERS_URLS = [
    {
        'name': 'faers_ascii_2024q3',
        'url': 'https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip'
    }
]

def download_faers():
    for item in FAERS_URLS:
        zip_path = os.path.join(RAW_DIR, f"{item['name']}.zip")
        extract_path = os.path.join(RAW_DIR, item['name'])

        print(f"📥 다운로드 중: {item['name']}")

        response = requests.get(item['url'], stream=True)
        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total * 100) if total else 0
                print(f"\r  진행률: {percent:.1f}%", end='', flush=True)

        print(f"\n✅ 다운로드 완료: {zip_path}")

        # 압축 해제
        print(f"📦 압축 해제 중...") 
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ 압축 해제 완료: {extract_path}")

if __name__ == "__main__":
    download_faers()
    
       