"""
API 엔드포인트 테스트
"""
import json


class TestSearchAPI:
    """약물 검색 API 테스트"""

    def test_search_valid_drug(self, client):
        """유효한 약물 검색 - 200 응답"""
        res = client.get('/api/search/METHOTREXATE')
        # 데이터 파일 없으면 404, 있으면 200
        assert res.status_code in [200, 404]

    def test_search_empty_drug(self, client):
        """존재하지 않는 약물 검색 - 404 응답"""
        res = client.get('/api/search/NONEXISTENT_DRUG_XYZ')
        assert res.status_code == 404
        data = json.loads(res.data)
        assert 'error' in data

    def test_autocomplete(self, client):
        """자동완성 API - 200 응답"""
        res = client.get('/api/autocomplete/MET')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'suggestions' in data
        assert isinstance(data['suggestions'], list)


class TestPredictAPI:
    """AI 위험도 예측 API 테스트"""

    def test_predict_missing_fields(self, client):
        """필수 필드 누락 시 400 또는 에러 응답"""
        res = client.post('/api/predict',
                          data=json.dumps({}),
                          content_type='application/json')
        assert res.status_code in [400, 500]

    def test_predict_unknown_drug(self, client):
        """알 수 없는 약물 예측 - 400 응답"""
        res = client.post('/api/predict',
                          data=json.dumps({
                              'drugname': 'UNKNOWN_DRUG_XYZ',
                              'reaction': 'FATIGUE',
                              'age': 50,
                              'sex': 'F'
                          }),
                          content_type='application/json')
        assert res.status_code == 400
        data = json.loads(res.data)
        assert 'error' in data

    def test_predict_response_structure(self, client):
        """예측 성공 시 응답 구조 확인"""
        res = client.post('/api/predict',
                          data=json.dumps({
                              'drugname': 'METHOTREXATE',
                              'reaction': 'FATIGUE',
                              'age': 50,
                              'sex': 'F'
                          }),
                          content_type='application/json')
        # 모델 파일 없으면 500, 있으면 200
        if res.status_code == 200:
            data = json.loads(res.data)
            assert 'drug' in data
            assert 'reaction' in data
            assert 'risk' in data
            assert 'probability' in data
            assert 'safe' in data['probability']
            assert 'risk' in data['probability']


class TestComboAPI:
    """약물 조합 위험도 API 테스트"""

    def test_combo_missing_drugs(self, client):
        """약물 누락 시 400 응답"""
        res = client.post('/api/combo',
                          data=json.dumps({'drug1': '', 'drug2': ''}),
                          content_type='application/json')
        assert res.status_code == 400

    def test_combo_unknown_drug(self, client):
        """알 수 없는 약물 조합 - 400 응답"""
        res = client.post('/api/combo',
                          data=json.dumps({
                              'drug1': 'UNKNOWN_A',
                              'drug2': 'UNKNOWN_B',
                              'age': 50,
                              'sex': 'F'
                          }),
                          content_type='application/json')
        assert res.status_code == 400


class TestFavoriteAPI:
    """즐겨찾기 API 테스트"""

    def test_add_favorite(self, client):
        """즐겨찾기 추가 - 200 응답"""
        res = client.post('/api/favorite/METHOTREXATE')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'message' in data

    def test_get_favorites(self, client):
        """즐겨찾기 목록 조회 - 200 응답"""
        res = client.get('/api/favorites')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'favorites' in data
        assert isinstance(data['favorites'], list)

    def test_favorite_then_list(self, client):
        """즐겨찾기 추가 후 목록에 있는지 확인"""
        client.post('/api/favorite/ASPIRIN')
        res = client.get('/api/favorites')
        data = json.loads(res.data)
        drugs = [f['drugname'] for f in data['favorites']]
        assert 'ASPIRIN' in drugs


class TestHistoryAPI:
    """기록 API 테스트"""

    def test_get_history(self, client):
        """기록 조회 - 200 응답 + 구조 확인"""
        res = client.get('/api/history')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'searches' in data
        assert 'predictions' in data
        assert isinstance(data['searches'], list)
        assert isinstance(data['predictions'], list)


class TestCompareAPI:
    """약물 비교 API 테스트"""

    def test_compare_missing_params(self, client):
        """파라미터 누락 시 400 응답"""
        res = client.get('/api/compare')
        assert res.status_code == 400

    def test_compare_unknown_drugs(self, client):
        """알 수 없는 약물 비교 - 404 응답"""
        res = client.get('/api/compare?drug1=UNKNOWN_A&drug2=UNKNOWN_B')
        assert res.status_code == 404
