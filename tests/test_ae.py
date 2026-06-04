"""
AE Manager 모듈 테스트
CTCAE 자동 분류, SAE 판정, 보고 타임라인 검증
"""
import json
from datetime import datetime, timedelta

from tests.conftest import client


class TestAEAutoClassify:
    """CTCAE 자동 분류 로직 테스트"""

    def test_ae_list_empty(self, client):
        """AE 목록 API - 200 응답 + 구조 확인"""
        res = client.get('/api/ae/list')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'reports' in data
        assert 'summary' in data
        assert isinstance(data['reports'], list)
        assert 'total' in data['summary']

    def test_create_ae_missing_fields(self, client):
        """필수 필드 누락 시 400 응답"""
        res = client.post('/api/ae/create',
                          data=json.dumps({}),
                          content_type='application/json')
        assert res.status_code == 400
        data = json.loads(res.data)
        assert 'error' in data

    def test_create_normal_ae(self, client):
        """일반 AE 등록 - Grade 1, SAE 아님"""
        res = client.post('/api/ae/create',
                          data=json.dumps({
                              'patient_code': 'PT-001',
                              'drugname': 'ASPIRIN',
                              'ae_term': 'NAUSEA',
                              'age': 45,
                              'sex': 'F',
                              'causality': 'Possible',
                              'outcome': '회복'
                          }),
                          content_type='application/json')
        assert res.status_code == 201
        data = json.loads(res.data)
        assert data['ctcae_grade'] == 1
        assert data['is_sae'] == False
        assert data['report_deadline'] is None   # 일반 AE는 마감일 없음

    def test_create_sae_hospitalization(self, client):
        """입원 AE 등록 - Grade 3, SAE 해당, 15일 마감일 설정"""
        res = client.post('/api/ae/create',
                          data=json.dumps({
                              'patient_code': 'PT-002',
                              'drugname': 'METHOTREXATE',
                              'ae_term': 'HOSPITALIZATION',
                              'age': 60,
                              'sex': 'M',
                              'causality': 'Probable',
                              'outcome': '회복중'
                          }),
                          content_type='application/json')
        assert res.status_code == 201
        data = json.loads(res.data)
        assert data['ctcae_grade'] == 3
        assert data['is_sae'] == True
        assert data['report_deadline'] is not None  # SAE는 마감일 있음

    def test_create_sae_death(self, client):
        """사망 AE 등록 - Grade 5, SAE 해당"""
        res = client.post('/api/ae/create',
                          data=json.dumps({
                              'patient_code': 'PT-003',
                              'drugname': 'WARFARIN',
                              'ae_term': 'DEATH',
                              'age': 75,
                              'sex': 'M',
                          }),
                          content_type='application/json')
        assert res.status_code == 201
        data = json.loads(res.data)
        assert data['ctcae_grade'] == 5
        assert data['is_sae'] == True

    def test_create_ae_manual_grade_override(self, client):
        """Grade 직접 입력 시 자동 판정보다 우선"""
        res = client.post('/api/ae/create',
                          data=json.dumps({
                              'patient_code': 'PT-004',
                              'drugname': 'IBUPROFEN',
                              'ae_term': 'HEADACHE',
                              'ctcae_grade': 3,   # 직접 Grade 3 입력
                              'age': 30,
                              'sex': 'F',
                          }),
                          content_type='application/json')
        assert res.status_code == 201
        data = json.loads(res.data)
        assert data['ctcae_grade'] == 3  # 직접 입력값 반영


class TestAECRUD:
    """AE CRUD 테스트"""

    def _create_sample_ae(self, client, patient='PT-TEST', ae_term='NAUSEA'):
        """테스트용 AE 생성 헬퍼"""
        res = client.post('/api/ae/create',
                          data=json.dumps({
                              'patient_code': patient,
                              'drugname': 'ASPIRIN',
                              'ae_term': ae_term,
                          }),
                          content_type='application/json')
        return json.loads(res.data)

    def test_get_ae_detail(self, client):
        """AE 상세 조회"""
        created = self._create_sample_ae(client)
        ae_id = created['id']
        res = client.get(f'/api/ae/{ae_id}')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['id'] == ae_id
        assert data['patient_code'] == 'PT-TEST'

    def test_get_ae_not_found(self, client):
        """존재하지 않는 AE 조회 - 404"""
        res = client.get('/api/ae/99999')
        assert res.status_code == 404

    def test_submit_ae(self, client):
        """AE 제출 완료 처리"""
        created = self._create_sample_ae(client)
        ae_id = created['id']
        res = client.post(f'/api/ae/{ae_id}/submit')
        assert res.status_code == 200
        # 제출 후 상세 조회해서 is_submitted 확인
        detail = json.loads(client.get(f'/api/ae/{ae_id}').data)
        assert detail['is_submitted'] == True

    def test_delete_ae(self, client):
        """AE 삭제"""
        created = self._create_sample_ae(client, patient='PT-DEL')
        ae_id = created['id']
        res = client.post(f'/api/ae/{ae_id}/delete')
        assert res.status_code == 200
        # 삭제 후 조회 시 404
        res2 = client.get(f'/api/ae/{ae_id}')
        assert res2.status_code == 404

    def test_ae_list_after_create(self, client):
        """AE 등록 후 목록에 반영되는지 확인"""
        self._create_sample_ae(client, patient='PT-LIST1')
        self._create_sample_ae(client, patient='PT-LIST2')
        res = client.get('/api/ae/list')
        data = json.loads(res.data)
        assert data['summary']['total'] == 2

    def test_sae_filter(self, client):
        """SAE 필터링 테스트"""
        # 일반 AE 1개
        self._create_sample_ae(client, patient='PT-AE', ae_term='NAUSEA')
        # SAE 1개
        self._create_sample_ae(client, patient='PT-SAE', ae_term='HOSPITALIZATION')

        res = client.get('/api/ae/list?sae_only=true')
        data = json.loads(res.data)
        assert all(r['is_sae'] for r in data['reports'])


class TestAEStats:
    """AE 통계 테스트"""

    def test_stats_empty(self, client):
        """AE 없을 때 stats 응답"""
        res = client.get('/api/ae/stats')
        assert res.status_code == 200

    def test_stats_after_create(self, client):
        """AE 등록 후 통계 확인"""
        # AE 2개 등록
        client.post('/api/ae/create',
                    data=json.dumps({'patient_code': 'PT-S1', 'drugname': 'ASPIRIN', 'ae_term': 'NAUSEA'}),
                    content_type='application/json')
        client.post('/api/ae/create',
                    data=json.dumps({'patient_code': 'PT-S2', 'drugname': 'METHOTREXATE', 'ae_term': 'HOSPITALIZATION'}),
                    content_type='application/json')

        res = client.get('/api/ae/stats')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['total'] == 2
        assert data['sae_count'] == 1
        assert 'grade_distribution' in data
