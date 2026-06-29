"""
SHAP 기반 AI 예측 설명 엔드포인트 (/api/shap/explain) 테스트
Windows에서 XGBoost+SHAP C 확장이 힙 손상을 일으키는 알려진 버그로 인해
compute_shap을 mock으로 대체하여 API 레이어만 테스트합니다.
"""
import json
import pytest
from unittest.mock import patch

MOCK_SHAP_RESULT = {
    'drug': 'METHOTREXATE',
    'reaction': 'FATIGUE',
    'prediction': 1,
    'risk_label': 'HIGH RISK',
    'probability': {'safe': 30.0, 'risk': 70.0},
    'shap': [
        {'feature': 'combo_risk_rate', 'value': 0.8,          'shap': 0.412},
        {'feature': 'drug_risk_rate',  'value': 0.75,         'shap': 0.305},
        {'feature': 'drug',            'value': 'METHOTREXATE','shap': 0.198},
        {'feature': 'reaction',        'value': 'FATIGUE',    'shap': 0.12},
        {'feature': 'age',             'value': 50,           'shap': -0.05},
        {'feature': 'sex',             'value': 'F',          'shap': -0.02},
        {'feature': 'reac_risk_rate',  'value': 0.6,          'shap': 0.01},
    ]
}

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
        """
        Windows XGBoost+SHAP C 확장 힙 손상 버그 우회:
        compute_shap을 mock으로 대체하고 API 응답 구조만 검증.
        """
        with patch('app.routes.analysis.shap_xai.compute_shap', return_value=MOCK_SHAP_RESULT):
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
        """feature 이름이 한국어로 변환됐는지 확인 (raw 키가 아닌지 검증)"""
        with patch('app.routes.analysis.shap_xai.compute_shap', return_value=MOCK_SHAP_RESULT):
            res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        data = json.loads(res.data)
        raw_keys = {'drug', 'reaction', 'sex', 'age', 'drug_risk_rate', 'reac_risk_rate', 'combo_risk_rate'}
        for f in data['top_features']:
            assert f['feature'] not in raw_keys

    def test_explanation_has_no_chinese_or_devanagari(self, client):
        """Ollama 응답에 중국어/데바나가리 등 외국어 문자가 없는지 확인"""
        import re
        with patch('app.routes.analysis.shap_xai.compute_shap', return_value=MOCK_SHAP_RESULT):
            res = post(client, {'drug': 'METHOTREXATE', 'reaction': 'FATIGUE', 'age': 50, 'sex': 'F'})
        data = json.loads(res.data)
        assert not re.search(r'[\u4E00-\u9FFF\u3400-\u4DBF\u0900-\u097F]', data['explanation'])

