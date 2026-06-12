"""
용량 계산 API (CrCl, 소아 용량, BSA) 테스트
- 데이터 파일(FAERS CSV)에 의존하지 않는 순수 계산 로직
"""
import json
import math


def post(client, url, payload):
    return client.post(url, data=json.dumps(payload), content_type='application/json')


class TestCrCl:
    """크레아티닌 청소율 (Cockcroft-Gault) 계산"""

    def test_male_normal_renal_function(self, client):
        res = post(client, '/api/dosage/crcl', {'age': 40, 'weight': 70, 'creatinine': 1.0, 'sex': 'M'})
        assert res.status_code == 200
        data = json.loads(res.data)
        expected = ((140 - 40) * 70) / (72 * 1.0)
        assert data['crcl'] == round(expected, 1)
        assert data['stage'] == 'Normal (G1)'
        assert data['color'] == 'green'

    def test_female_applies_correction_factor(self, client):
        res_m = post(client, '/api/dosage/crcl', {'age': 50, 'weight': 60, 'creatinine': 1.0, 'sex': 'M'})
        res_f = post(client, '/api/dosage/crcl', {'age': 50, 'weight': 60, 'creatinine': 1.0, 'sex': 'F'})
        crcl_m = json.loads(res_m.data)['crcl']
        crcl_f = json.loads(res_f.data)['crcl']
        assert crcl_f == round(crcl_m * 0.85, 1)

    def test_severe_renal_impairment_stage(self, client):
        # 매우 낮은 청소율이 나오도록 노인+저체중+고크레아티닌
        res = post(client, '/api/dosage/crcl', {'age': 85, 'weight': 40, 'creatinine': 8.0, 'sex': 'F'})
        data = json.loads(res.data)
        assert data['crcl'] < 15
        assert data['stage'] == 'Renal failure (G5)'
        assert data['color'] == 'darkred'

    def test_missing_fields_returns_400(self, client):
        res = post(client, '/api/dosage/crcl', {'age': 40, 'weight': 70})
        assert res.status_code == 400

    def test_zero_age_returns_400(self, client):
        res = post(client, '/api/dosage/crcl', {'age': 0, 'weight': 70, 'creatinine': 1.0})
        assert res.status_code == 400


class TestPediatricDosage:
    """소아 용량 계산 (Clark's rule, Young's rule, BSA)"""

    def test_clark_rule(self, client):
        res = post(client, '/api/dosage/pediatric', {'adult_dose': 100, 'weight': 35})
        data = json.loads(res.data)
        assert data['clark'] == round(100 * 35 / 70, 2)

    def test_young_rule(self, client):
        res = post(client, '/api/dosage/pediatric', {'adult_dose': 100, 'age': 6})
        data = json.loads(res.data)
        assert data['young'] == round(100 * 6 / (6 + 12), 2)

    def test_bsa_rule_requires_weight_and_height(self, client):
        res = post(client, '/api/dosage/pediatric', {'adult_dose': 100, 'weight': 20, 'height': 110})
        data = json.loads(res.data)
        expected_bsa = math.sqrt((110 * 20) / 3600)
        assert data['bsa_value'] == round(expected_bsa, 2)
        assert data['bsa'] == round(100 * expected_bsa / 1.73, 2)

    def test_only_adult_dose_returns_empty_results(self, client):
        res = post(client, '/api/dosage/pediatric', {'adult_dose': 100})
        data = json.loads(res.data)
        assert data == {}

    def test_missing_adult_dose_returns_400(self, client):
        res = post(client, '/api/dosage/pediatric', {'weight': 20, 'age': 5})
        assert res.status_code == 400


class TestBSA:
    """체표면적 (Mosteller / Du Bois) 계산"""

    def test_mosteller_and_dubois(self, client):
        res = post(client, '/api/dosage/bsa', {'weight': 70, 'height': 170})
        assert res.status_code == 200
        data = json.loads(res.data)
        expected_mosteller = math.sqrt((170 * 70) / 3600)
        expected_dubois = 0.007184 * (170 ** 0.725) * (70 ** 0.425)
        assert data['bsa_mosteller'] == round(expected_mosteller, 3)
        assert data['bsa_dubois'] == round(expected_dubois, 3)
        assert data['total_dose'] is None

    def test_total_dose_with_dose_per_m2(self, client):
        res = post(client, '/api/dosage/bsa', {'weight': 70, 'height': 170, 'dose_per_m2': 100})
        data = json.loads(res.data)
        bsa = math.sqrt((170 * 70) / 3600)
        assert data['total_dose'] == round(bsa * 100, 2)

    def test_missing_height_returns_400(self, client):
        res = post(client, '/api/dosage/bsa', {'weight': 70})
        assert res.status_code == 400
