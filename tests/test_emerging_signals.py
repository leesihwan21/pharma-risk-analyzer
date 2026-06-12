"""
분기별 PRR 비교 기반 신규/이상 신호 탐지 (/api/signals/emerging) 테스트
"""
import json
from app.routes.analysis import compute_emerging_signals


class TestEmergingSignals:
    def test_unknown_drug_returns_404(self, client):
        res = client.get('/api/signals/emerging/UNKNOWN_DRUG_XYZ')
        assert res.status_code == 404

    def test_known_drug_returns_structure(self, client):
        res = client.get('/api/signals/emerging/METHOTREXATE')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drugname'] == 'METHOTREXATE'
        assert 'latest_quarter' in data
        assert isinstance(data['emerging'], list)

    def test_emerging_entries_meet_signal_criteria(self, client):
        """emerging에 포함된 항목은 최신 분기 PRR>=2, 건수>=3, 이전 분기엔 비신호였어야 함"""
        res = client.get('/api/signals/emerging/METHOTREXATE')
        data = json.loads(res.data)
        for e in data['emerging']:
            assert e['prr_latest'] >= 2
            assert e['prr_history'] < 2
            assert e['latest_count'] >= 3
            assert 'reaction' in e
            assert 'quarter' in e

    def test_lowercase_drugname_normalized(self, client):
        res = client.get('/api/signals/emerging/methotrexate')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drugname'] == 'METHOTREXATE'


class TestFavoritesAlertsEmergingIntegration:
    def test_alert_level_consistent_with_emerging(self, client):
        client.post('/api/favorite/METHOTREXATE')
        res = client.get('/api/favorites/alerts')
        data = json.loads(res.data)

        alert = next(a for a in data['alerts'] if a['drugname'] == 'METHOTREXATE')
        emerging = compute_emerging_signals('METHOTREXATE')
        has_emerging = bool(emerging and emerging['emerging'])

        if has_emerging:
            assert alert['level'] == 'new'
            assert len(alert['emerging_signals']) > 0
        else:
            assert alert['level'] != 'new'
            assert alert['emerging_signals'] == []
