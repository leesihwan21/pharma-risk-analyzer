import os
from dotenv import load_dotenv
from app.dur_lookup import check_combo_dur_taboo, query_dur_taboo, get_dur_mapping

load_dotenv(os.path.join('app', '.env'))
API_KEY = os.environ.get('MFDS_API_KEY', '')

print(f"[API_KEY 확인] {'있음' if API_KEY else '없음! .env 확인 필요'}")
print()

# 1. 매핑 테이블 확인
print("=== 매핑 테이블 확인 ===")
print("METHOTREXATE ->", get_dur_mapping("METHOTREXATE"))
print("DICLOFENAC SODIUM ->", get_dur_mapping("DICLOFENAC SODIUM"))
print()

# 2. 단일 약물 DUR 병용금기 raw 조회 (전체 건수 + 고유 병용금기 대상 성분 목록 확인)
print("=== METHOTREXATE 병용금기 raw 조회 ===")
items = query_dur_taboo("메토트렉세이트", API_KEY)
print(f"총 {len(items)}건 조회됨")
unique_mixtures = set()
for it in items:
    unique_mixtures.add(it.get("MIXTURE_INGR_ENG_NAME", "?"))
print("병용금기 대상 고유 성분 목록:", unique_mixtures)
print()

# 3. 두 약물 조합 DUR 체크
print("=== METHOTREXATE + ASPIRIN 조합 체크 (실제 양성 사례 확인용) ===")
result_pos = check_combo_dur_taboo("METHOTREXATE", "ASPIRIN", API_KEY)
print(result_pos)
print()

print("=== METHOTREXATE + DICLOFENAC SODIUM 조합 체크 ===")
result = check_combo_dur_taboo("METHOTREXATE", "DICLOFENAC SODIUM", API_KEY)
print(result)
print()

# 4. 생물학적제제 조합 체크 (DUR 대상 아님 케이스 확인)
print("=== HUMIRA + METHOTREXATE 조합 체크 (생물학적제제 케이스) ===")
result2 = check_combo_dur_taboo("HUMIRA", "METHOTREXATE", API_KEY)
print(result2)
