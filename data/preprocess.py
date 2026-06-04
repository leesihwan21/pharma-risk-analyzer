import pandas as pd
import os

ASCII_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'raw', 'faers_ascii_2024q3', 'ASCII')

def load_data():
    print("📂 데이터 로딩 중...")

    # 환자 정보
    demo = pd.read_csv(
        os.path.join(ASCII_DIR, 'DEMO24Q3.txt'),
        sep='$',
        encoding='latin-1',
        low_memory=False
    )

    # 약물 정보
    drug = pd.read_csv(
        os.path.join(ASCII_DIR, 'DRUG24Q3.txt'),
        sep='$',
        encoding='latin-1',
        low_memory=False
    )

    # 부작용 정보
    reac = pd.read_csv(
        os.path.join(ASCII_DIR, 'REAC24Q3.txt'),
        sep='$',
        encoding='latin-1',
        low_memory=False
    )

    # 결과
    outc = pd.read_csv(
        os.path.join(ASCII_DIR, 'OUTC24Q3.txt'),
        sep='$',
        encoding='latin-1',
        low_memory=False
    )

    print(f"✅ DEMO: {demo.shape}")
    print(f"✅ DRUG: {drug.shape}")
    print(f"✅ REAC: {reac.shape}")
    print(f"✅ OUTC: {outc.shape}")

    return demo, drug, reac, outc

def preprocess(demo, drug, reac, outc):
    print("DEMO 컬럼:", demo.columns.tolist())
    print("DRUG 컬럼:", drug.columns.tolist())
    print("REAC 컬럼:", reac.columns.tolist())
    print("OUTC 컬럼:", outc.columns.tolist())

def preprocess_data(demo, drug, reac, outc):
    print("🔄 데이터 전처리 중...")

    # 환자 정보에서 필요한 컬럼만 선택
    demo = demo[['primaryid', 'age', 'sex', 'reporter_country']]
    drug = drug[['primaryid', 'drugname', 'role_cod']]
    reac = reac[['primaryid', 'pt']]
    outc = outc[['primaryid', 'outc_cod']]

    # 컬럼 소문자 통일
    demo.columns = demo.columns.str.lower()
    drug.columns = drug.columns.str.lower()
    reac.columns = reac.columns.str.lower()
    outc.columns = outc.columns.str.lower()

    # 약물명 대문자 통일 및 결측치 제거
    drug['drugname'] = drug['drugname'].str.upper().str.strip()
    drug = drug.dropna(subset=['drugname'])

    # 부작용명 대문자 통일 및 결측치 제거
    reac['pt'] = reac['pt'].str.upper().str.strip()
    reac = reac.dropna(subset=['pt'])

    # 나이 이상값 제거 (0~120세 사이)
    if 'age' in demo.columns:
        demo = demo[(demo['age'] >= 0) & (demo['age'] <= 120)]

    print("✅ 데이터 전처리 완료")
    return demo, drug, reac, outc


def merge_and_save(demo, drug, reac, outc):
    print("\n🔗 데이터 병합 중...")

    # processed 폴더 생성
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    os.makedirs(save_dir, exist_ok=True)

    # drug + reac 병합
    drug_reac = pd.merge(drug[['primaryid', 'drugname']],
                         reac[['primaryid', 'pt']],
                         on='primaryid', how='inner')

    # demo 병합
    merged = pd.merge(drug_reac,
                      demo[['primaryid', 'age', 'sex', 'reporter_country']],
                      on='primaryid', how='left')

    # outc 병합
    merged = pd.merge(merged,
                      outc[['primaryid', 'outc_cod']],
                      on='primaryid', how='left')

    # 샘플링 (50만 행으로 제한)
    if len(merged) > 500000:
        merged = merged.sample(n=500000, random_state=42)
        print(f"⚠️  데이터 샘플링 적용: 500,000행")

    print(f"✅ 병합 완료: {merged.shape}")

    save_path = os.path.join(save_dir, 'processed_faers.csv')
    merged.to_csv(save_path, index=False)
    print(f"💾 저장 완료: {save_path}")

    return merged



if __name__ == "__main__":
    demo, drug, reac, outc = load_data()
    demo, drug, reac, outc = preprocess_data(demo, drug, reac, outc)
    merged_data = merge_and_save(demo, drug, reac, outc)
    print("🎉 데이터 전처리 및 병합 완료!")
    print(df.head())