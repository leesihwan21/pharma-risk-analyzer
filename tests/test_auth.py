"""
인증(회원가입/로그인/로그아웃) 및 관리자 권한 테스트
"""
import json


def register(client, username='tester', email='tester@example.com', password='pass1234'):
    return client.post('/register',
                        data=json.dumps({'username': username, 'email': email, 'password': password}),
                        content_type='application/json')


def login(client, username='tester', password='pass1234'):
    return client.post('/login',
                        data=json.dumps({'username': username, 'password': password}),
                        content_type='application/json')


class TestRegister:
    def test_register_success(self, client):
        res = register(client)
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['username'] == 'tester'
        assert data['role'] == 'USER' or data['role'] == 'User'

    def test_register_duplicate_username(self, client):
        register(client)
        res = register(client, email='other@example.com')
        assert res.status_code == 400

    def test_register_duplicate_email(self, client):
        register(client)
        res = register(client, username='tester2')
        assert res.status_code == 400


class TestLogin:
    def test_login_success(self, client):
        register(client)
        client.get('/logout')
        res = login(client)
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data['username'] == 'tester'

    def test_login_wrong_password(self, client):
        register(client)
        client.get('/logout')
        res = login(client, password='wrongpass')
        assert res.status_code == 401

    def test_login_unknown_user(self, client):
        res = login(client, username='nouser')
        assert res.status_code == 401


class TestSession:
    def test_me_logged_in(self, client):
        register(client)
        res = client.get('/api/me')
        data = json.loads(res.data)
        assert data['logged_in'] is True
        assert data['username'] == 'tester'

    def test_me_logged_out(self, client):
        res = client.get('/api/me')
        data = json.loads(res.data)
        assert data['logged_in'] is False

    def test_logout(self, client):
        register(client)
        res = client.get('/logout')
        assert res.status_code == 200
        data = json.loads(client.get('/api/me').data)
        assert data['logged_in'] is False


class TestAdminAccessControl:
    """일반 유저는 관리자 페이지/API에 접근할 수 없어야 함"""

    def test_regular_user_cannot_access_admin_page(self, client):
        register(client)
        res = client.get('/admin')
        assert res.status_code == 403

    def test_regular_user_cannot_access_admin_users_api(self, client):
        register(client)
        res = client.get('/api/admin/users')
        assert res.status_code == 403

    def test_regular_user_cannot_access_admin_logs_api(self, client):
        register(client)
        res = client.get('/api/admin/logs')
        assert res.status_code == 403

    def test_logged_out_admin_page_requires_login(self, client):
        res = client.get('/admin')
        # flask-login redirects unauthenticated users to login page
        assert res.status_code in (302, 401, 403)

    def test_make_admin_blocked_for_other_users(self, client):
        """make-admin은 지정된 본인 계정 외에는 거부해야 함"""
        register(client)
        res = client.get('/make-admin')
        assert res.status_code == 403

    def test_make_admin_does_not_grant_admin_access(self, client):
        register(client)
        client.get('/make-admin')
        res = client.get('/api/admin/users')
        assert res.status_code == 403
