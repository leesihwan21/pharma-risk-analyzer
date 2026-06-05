import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# ✅ 쿼터별 파일명 패턴 정의
def quarter_to_prefix(quarter: str) -> str:
    """'2024q3' → '24Q3'"""
    year, q = quarter.split('q')
    return f'{year[2:]}Q{q}'


def load_and_merge_quarter(quarter: str) -> pd.DataFrame:
    """
    단일 쿼터 로드 + 병합 + 샘플링까지 한 번에 처리
    → 메모리에 쿼터 하나치만 올라감
    """
    prefix = quarter_to_prefix(quarter)
    ascii_dir = os.path.join(RAW_DIR, f'faers_ascii_{quarter}', 'ASCII')

    print(f'  📂 {quarter} 로딩 중...')

    # 필요한 컬럼만 읽어서 메모리 절약
    demo = pd.read_csv(
        os.path.join(ascii_dir, f'DEMO{prefix}.txt'),
        sep='$', encoding='latin-1', low_memory=False,
        usecols=lambda c: c.strip().lower() in ['primaryid', 'age', 'sex', 'reporter_country']
    )
    drug = pd.read_csv(
        os.path.join(ascii_dir, f'DRUG{prefix}.txt'),
        sep='$', encoding='latin-1', low_memory=False,
        usecols=lambda c: c.strip().lower() in ['primaryid', 'drugname', 'role_cod']
    )
    reac = pd.read_csv(
        os.path.join(ascii_dir, f'REAC{prefix}.txt'),
        sep='$', encoding='latin-1', low_memory=False,
        usecols=lambda c: c.strip().lower() in ['primaryid', 'pt']
    )
    outc = pd.read_csv(
        os.path.join(ascii_dir, f'OUTC{prefix}.txt'),
        sep='$', encoding='latin-1', low_memory=False,
        usecols=lambda c: c.strip().lower() in ['primaryid', 'outc_cod']
    )

    # 컬럼 소문자 통일
    for df in [demo, drug, reac, outc]:
        df.columns = df.columns.str.strip().str.lower()

    # 정제
    drug['drugname'] = drug['drugname'].str.upper().str.strip()
    drug = drug.dropna(subset=['drugname'])
    reac['pt'] = reac['pt'].str.upper().str.strip()
    reac = reac.dropna(subset=['pt'])
    demo = demo[(demo['age'] >= 0) & (demo['age'] <= 120)]

    # 병합
    merged = pd.merge(drug[['primaryid', 'drugname']], reac[['primaryid', 'pt']], on='primaryid', how='inner')
    merged = pd.merge(merged, demo[['primaryid', 'age', 'sex', 'reporter_country']], on='primaryid', how='left')
    merged = pd.merge(merged, outc[['primaryid', 'outc_cod']], on='primaryid', how='left')

    # ✅ 쿼터 컬럼 추가 — 시계열 분석의 핵심
    merged['quarter'] = quarter

    # 쿼터별 샘플링 (12만 행) — 4개 쿼터 × 12만 = 최대 48만행
    SAMPLE_PER_QUARTER = 120000
    if len(merged) > SAMPLE_PER_QUARTER:
        merged = merged.sample(n=SAMPLE_PER_QUARTER, random_state=42)
        print(f'  ⚠️  샘플링 적용: {SAMPLE_PER_QUARTER:,}행')

    print(f'  ✅ {quarter} 완료 — {len(merged):,}행')
    return merged


def process_all_quarters(quarters: list) -> pd.DataFrame:
    """쿼터별로 처리 후 합치기 — 메모리 효율적"""
    results = []

    for quarter in quarters:
        ascii_dir = os.path.join(RAW_DIR, f'faers_ascii_{quarter}', 'ASCII')
        if not os.path.exists(ascii_dir):
            print(f'  ⚠️  {quarter} 데이터 없음, 스킵')
            continue
        df = load_and_merge_quarter(quarter)
        results.append(df)

    print(f'\n🔗 {len(results)}개 쿼터 합치는 중...')
    final = pd.concat(results, ignore_index=True)
    print(f'✅ 최종 데이터: {final.shape}')
    print(f'   포함된 쿼터: {sorted(final["quarter"].unique())}')
    return final


def save(merged: pd.DataFrame):
    save_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'processed_faers.csv')
    merged.to_csv(save_path, index=False)
    print(f'💾 저장 완료: {save_path}')


if __name__ == '__main__':
    # ✅ 처리할 쿼터 목록
    QUARTERS = [
        '2024q1',
        '2024q2',
        '2024q3',
        '2025q1',
    ]

    merged = process_all_quarters(QUARTERS)
    save(merged)
    print('\n🎉 전체 완료!')
    print(merged.head())
