"""
SHAP 기반 AI 자연어 설명 (/api/shap/explain) 테스트
"""
import json


def post(client, payload):
    return client.post('/api/shap/explain', data=json.dumps(payload), content_type='application/json')


class TestShapExplain:
    def test_unknown_drug_returns_400(self, client):
        res = post(client, {'drug': 'UNKNOWN_XYZ', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        assert res.status_code == 400

    def test_unknown_reaction_returns_400(self, client):
        res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'UNKNOWN_XYZ', 'age': 50, 'sex': 'F'})
        assert res.status_code == 400

    def test_valid_input_returns_top_features_and_explanation(self, client):
        res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['drug'] == 'METHOTREXATE'
        assert data['reaction'] == 'FATIGUE'
        assert 'explanation' in data
        assert isinstance(data['explanation'], str)
        assert len(data['top_features']) == 3
        for f in data['top_features']:
            assert 'feature' in f
            assert 'value' in f
            assert 'shap' in f

    def test_top_features_are_korean_labels(self, client):
        """feature 이름이 한국어 라벨로 변환되어야 함 (drug_risk_rate 등 영문 키 노출 방지)"""
        res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        data = json.loads(res.data)
        raw_keys = {'drug', 'reaction', 'sex', 'age', 'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate'}
        for f in data['top_features']:
            assert f['feature'] not in raw_keys

    def test_explanation_has_no_chinese_or_devanagari(self, client):
        """RAG와 동일하게 한자/힌디어 등 비한국어 문자가 섞이지 않아야 함"""
        import re
        res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        data = json.loads(res.data)
        assert not re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]', data['explanation'])
