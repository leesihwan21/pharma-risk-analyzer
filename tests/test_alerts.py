"""
즐겨찾기 약물 PRR 신호 알림 (약물감시) 테스트
- /api/favorites 와 동일하게 전역 즐겨찾기 기준 (로그인 불필요)
"""
import json


class TestFavoritesAlerts:
    def test_no_favorites_returns_empty_list(self, client):
        res = client.get('/api/favorites/alerts')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['alerts'] == []

    def test_favorite_drug_returns_alert_info(self, client):
        client.post('/api/favorite/METHOTREXATE')

        res = client.get('/api/favorites/alerts')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert len(data['alerts']) == 1

        alert = data['alerts'][0]
        assert alert['drugname'] == 'METHOTREXATE'
        assert alert['level'] in ('strong', 'signal', 'none')
        assert 'signal_count' in alert
        assert 'strong_signal_count' in alert
        assert 'top_signals' in alert

    def test_unknown_favorite_drug_is_skipped(self, client):
        # PRR 계산 가능한 데이터가 없는 약물 즐겨찾기
        client.post('/api/favorite/UNKNOWN_DRUG_XYZ')

        res = client.get('/api/favorites/alerts')
        assert res.status_code == 200
        data = json.loads(res.data)
        # 데이터에 없는 약물은 alerts에서 제외됨
        assert data['alerts'] == []

    def test_duplicate_favorites_deduplicated(self, client):
        client.post('/api/favorite/METHOTREXATE')
        client.post('/api/favorite/methotrexate')  # 대소문자 다른 중복

        res = client.get('/api/favorites/alerts')
        data = json.loads(res.data)
        drugnames = [a['drugname'] for a in data['alerts']]
        assert drugnames.count('METHOTREXATE') == 1

    def test_alerts_sorted_by_severity(self, client):
        client.post('/api/favorite/METHOTREXATE')
        client.post('/api/favorite/ASPIRIN')

        res = client.get('/api/favorites/alerts')
        data = json.loads(res.data)
        level_order = {'strong': 0, 'signal': 1, 'none': 2}
        levels = [level_order[a['level']] for a in data['alerts']]
        assert levels == sorted(levels)
