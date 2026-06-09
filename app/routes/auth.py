import os

from flask import Blueprint, render_template, jsonify, request
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import db, User
from app.models import db, User, UserActivityLog

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        email = data.get('email', '')
        password = data.get('password', '')

        if User.query.filter_by(username=username).first():
            return jsonify({'error': '이미 존재하는 사용자입니다.'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': '이미 존재하는 이메일입니다.'}), 400

        user = User(
            username=username,
            email=email,
            role='User',
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)

        log = UserActivityLog(user_id=user.id, username=username, action='register', ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
        return jsonify({'message': '회원가입 성공!', 'role': user.role, 'username': username})

    return render_template('register.html')

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': '아이디 또는 비밀번호가 틀렸습니다.'}), 401

        login_user(user)
        log  = UserActivityLog(user_id=user.id, username=username, action='login', ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
        return jsonify({'message': f'{username}님 로그인!', 'username': username})

    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    log = UserActivityLog(username=current_user.username, action='logout', ip_address=request.remote_addr)
    db.session.add(log)
    db.session.commit()
    return jsonify({'message': '로그아웃됐습니다.'})

@auth.route('/api/me')
def me():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

@auth.route('/my-role')
@login_required
def my_role():
    return jsonify({
        'username': current_user.username,
        'role': current_user.role
    })

@auth.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403

    users = User.query.all()
    return jsonify({
        'users': [
            {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M')
            }
            for user in users
        ]
    })

@auth.route('/make-admin')
@login_required
def make_admin():
    current_user.role = 'ADMIN'
    db.session.commit()
    return jsonify({
        'message': '관리자 변경 완료',
        'role': current_user.role
    })