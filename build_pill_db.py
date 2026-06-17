import os
import sqlite3
import time
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', '.env'))

API_KEY = os.environ.get('MFDS_API_KEY', '')
BASE_URL = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pill_identity.db')

print(f"[DEBUG] API_KEY 로드 확인: {'있음 (' + str(len(API_KEY)) + '자)' if API_KEY else '없음! .env 확인 필요'}")


def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pill_identity (
            item_seq TEXT PRIMARY KEY,
            item_name TEXT,
            entp_name TEXT,
            chart TEXT,
            drug_shape TEXT,
            color_class1 TEXT,
            color_class2 TEXT,
            print_front TEXT,
            print_back TEXT,
            line_front TEXT,
            line_back TEXT,
            item_image TEXT,
            class_name TEXT,
            form_code_name TEXT,
            etc_otc_name TEXT
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_print_front ON pill_identity(print_front)')
    conn.commit()


def fetch_page(page_no, num_of_rows=100):
    params = {
        'serviceKey': API_KEY,
        'pageNo': page_no,
        'numOfRows': num_of_rows,
        'type': 'json'
    }
    res = requests.get(BASE_URL, params=params, timeout=20)
    try:
        data = res.json()
    except Exception:
        print(f"[ERROR] JSON 파싱 실패. status={res.status_code}, raw={res.text[:300]}")
        raise
    body = data.get('body', {})
    return body.get('items', []), body.get('totalCount', 0)


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    page_no = 1
    num_of_rows = 100
    saved = 0

    while True:
        items, total_count = fetch_page(page_no, num_of_rows)
        if not items:
            break

        for item in items:
            conn.execute('''
                INSERT OR REPLACE INTO pill_identity
                (item_seq, item_name, entp_name, chart, drug_shape, color_class1, color_class2,
                 print_front, print_back, line_front, line_back, item_image, class_name,
                 form_code_name, etc_otc_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('ITEM_SEQ'), item.get('ITEM_NAME'), item.get('ENTP_NAME'),
                item.get('CHART'), item.get('DRUG_SHAPE'), item.get('COLOR_CLASS1'),
                item.get('COLOR_CLASS2'), item.get('PRINT_FRONT'), item.get('PRINT_BACK'),
                item.get('LINE_FRONT'), item.get('LINE_BACK'), item.get('ITEM_IMAGE'),
                item.get('CLASS_NAME'), item.get('FORM_CODE_NAME'), item.get('ETC_OTC_NAME')
            ))
        conn.commit()
        saved += len(items)
        print(f"[페이지 {page_no}] {saved}/{total_count} 저장 완료")

        if saved >= total_count:
            break
        page_no += 1
        time.sleep(0.2)

    conn.close()
    print(f"완료: 총 {saved}건 저장됨 -> {DB_PATH}")


if __name__ == '__main__':
    main()
