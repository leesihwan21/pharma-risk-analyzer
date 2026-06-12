"""
AE 리포트 PDF/ICH E2B(R3) XML 출력 및 감사 추적(Audit Trail) 테스트
"""
import json


def create_ae(client, **overrides):
    payload = {
        'patient_code': 'PT-001',
        'age': 45,
        'sex': 'F',
        'drugname': 'METHOTREXATE',
        'ae_term': 'NAUSEA',
        'ctcae_grade': 2,
        'outcome': '회복',
    }
    payload.update(overrides)
    res = client.post('/api/ae/create', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 201, res.data
    return json.loads(res.data)['id']


class TestAEPDF:
    def test_pdf_download(self, client):
        ae_id = create_ae(client)
        res = client.get(f'/api/ae/{ae_id}/pdf')
        assert res.status_code == 200
        assert res.mimetype == 'application/pdf'
        assert res.data[:4] == b'%PDF'

    def test_pdf_not_found(self, client):
        res = client.get('/api/ae/999999/pdf')
        assert res.status_code == 404

    def test_sae_pdf_includes_deadline(self, client):
        """SAE(입원) 리포트는 15일 보고 기한이 설정되어야 함"""
        ae_id = create_ae(client, sae_category='입원', ctcae_grade=4)
        detail = json.loads(client.get(f'/api/ae/{ae_id}').data)
        assert detail['is_sae'] is True
        assert detail['report_deadline'] is not None

        res = client.get(f'/api/ae/{ae_id}/pdf')
        assert res.status_code == 200
        assert res.mimetype == 'application/pdf'


class TestAEE2BExport:
    def test_e2b_xml_download(self, client):
        ae_id = create_ae(client)
        res = client.get(f'/api/ae/{ae_id}/e2b')
        assert res.status_code == 200
        assert res.mimetype == 'application/xml'
        body = res.data.decode('utf-8')
        assert '<?xml' in body
        assert 'ICH E2B' in body or 'ICSR' in body

    def test_e2b_contains_drug_and_patient_info(self, client):
        ae_id = create_ae(client, patient_code='PT-XYZ', drugname='ASPIRIN')
        res = client.get(f'/api/ae/{ae_id}/e2b')
        body = res.data.decode('utf-8')
        assert 'PT-XYZ' in body
        assert 'ASPIRIN' in body

    def test_e2b_not_found(self, client):
        res = client.get('/api/ae/999999/e2b')
        assert res.status_code == 404

    def test_e2b_sex_code_mapping(self, client):
        """성별 코드: 남성=1, 여성=2"""
        male_id = create_ae(client, patient_code='PT-M', sex='M')
        female_id = create_ae(client, patient_code='PT-F', sex='F')

        male_body = client.get(f'/api/ae/{male_id}/e2b').data.decode('utf-8')
        female_body = client.get(f'/api/ae/{female_id}/e2b').data.decode('utf-8')

        assert '<patientsex>1</patientsex>' in male_body
        assert '<patientsex>2</patientsex>' in female_body

    def test_e2b_outcome_code_recovered(self, client):
        ae_id = create_ae(client, outcome='회복')
        body = client.get(f'/api/ae/{ae_id}/e2b').data.decode('utf-8')
        assert '<reactionoutcome>1</reactionoutcome>' in body

    def test_e2b_outcome_code_fatal(self, client):
        ae_id = create_ae(client, outcome='사망', ctcae_grade=5)
        body = client.get(f'/api/ae/{ae_id}/e2b').data.decode('utf-8')
        assert '<reactionoutcome>5</reactionoutcome>' in body


class TestAuditTrail:
    def test_e2b_export_creates_audit_log(self, client):
        ae_id = create_ae(client)
        client.get(f'/api/ae/{ae_id}/e2b')
        res = client.get('/api/audit-trail')
        assert res.status_code == 200
        data = json.loads(res.data)
        actions = [l['action'] for l in data['logs']]
        assert 'EXPORT' in actions

    def test_audit_trail_returns_list(self, client):
        res = client.get('/api/audit-trail')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert isinstance(data['logs'], list)
