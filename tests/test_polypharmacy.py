"""
폴리파마시(다중 약물) 상호작용 체크 API 테스트
"""
import json


class TestPolypharmacyPage:
    def test_page_loads(self, client):
        res = client.get('/polypharmacy')
        assert res.status_code == 200


class TestPolypharmacyAPI:
    def test_requires_at_least_two_drugs(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE')
        assert res.status_code == 400

    def test_empty_drugs_param(self, client):
        res = client.get('/api/polypharmacy?drugs=')
        assert res.status_code == 400

    def test_rejects_more_than_five_drugs(self, client):
        drugs = 'METHOTREXATE,ASPIRIN,WARFARIN,METHOTREXATE2,ASPIRIN2,WARFARIN2'
        res = client.get('/api/polypharmacy?drugs=' + drugs)
        assert res.status_code == 400

    def test_unknown_drug_returns_404(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE,UNKNOWN_DRUG_XYZ')
        assert res.status_code == 404

    def test_two_known_drugs_returns_pairs(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE,ASPIRIN')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drugs'] == ['METHOTREXATE', 'ASPIRIN']
        assert len(data['pairs']) == 1
        pair = data['pairs'][0]
        assert pair['drug_a'] == 'METHOTREXATE'
        assert pair['drug_b'] == 'ASPIRIN'
        assert 'risk_score' in pair
        assert 'overall' in data
        assert 'co_occurrence' in data['overall']
        assert 'top_reactions' in data['overall']

    def test_three_drugs_returns_three_pairs(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE,ASPIRIN,WARFARIN')
        assert res.status_code == 200
        data = json.loads(res.data)
        # C(3,2) = 3 pairs
        assert len(data['pairs']) == 3
        pair_names = {(p['drug_a'], p['drug_b']) for p in data['pairs']}
        assert ('METHOTREXATE', 'ASPIRIN') in pair_names
        assert ('METHOTREXATE', 'WARFARIN') in pair_names
        assert ('ASPIRIN', 'WARFARIN') in pair_names

    def test_duplicate_drugs_are_deduplicated(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE,METHOTREXATE,ASPIRIN')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drugs'] == ['METHOTREXATE', 'ASPIRIN']
        assert len(data['pairs']) == 1

    def test_lowercase_input_normalized(self, client):
        res = client.get('/api/polypharmacy?drugs=methotrexate,aspirin')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drugs'] == ['METHOTREXATE', 'ASPIRIN']

    def test_high_risk_pairs_sorted_descending(self, client):
        res = client.get('/api/polypharmacy?drugs=METHOTREXATE,ASPIRIN,WARFARIN')
        data = json.loads(res.data)
        scores = [p['risk_score'] for p in data['high_risk_pairs']]
        assert scores == sorted(scores, reverse=True)
