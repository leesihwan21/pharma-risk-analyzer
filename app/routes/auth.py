import os
import secrets
from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, request, url_for
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from app.models import db, User, UserActivityLog, DrugSearch, PredictionLog, PasswordResetToken

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
        log = UserActivityLog(user_id=user.id, username=username, action='login', ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
        return jsonify({'message': f'{username}님 로그인!', 'username': username})

    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    log = UserActivityLog(user_id=current_user.id, username=current_user.username, action='logout', ip_address=request.remote_addr)
    db.session.add(log)
    db.session.commit()
    logout_user()
    return jsonify({'message': '로그아웃됐습니다.'})

@auth.route('/api/me')
def me():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username, 'role': current_user.role})
    return jsonify({'logged_in': False})

@auth.route('/my-role')
@login_required
def my_role():
    return jsonify({
        'username': current_user.username,
        'role': current_user.role
    })

@auth.route('/admin')
@login_required
def admin_page():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403
    return render_template('admin.html')

@auth.route('/api/admin/users')
@login_required
def admin_users():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({'users': [u.to_dict() for u in users]})

@auth.route('/api/admin/logs')
@login_required
def admin_logs():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403
    logs = UserActivityLog.query.order_by(UserActivityLog.created_at.desc()).limit(100).all()
    return jsonify({'logs': [l.to_dict() for l in logs]})

@auth.route('/api/admin/searches')
@login_required
def admin_searches():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403
    searches = DrugSearch.query.order_by(DrugSearch.searched_at.desc()).limit(100).all()
    return jsonify({'searches': [s.to_dict() for s in searches]})

@auth.route('/api/admin/predictions')
@login_required
def admin_predictions():
    if current_user.role != 'ADMIN':
        return jsonify({'error': '관리자 권한이 필요합니다.'}), 403
    predictions = PredictionLog.query.order_by(PredictionLog.predicted_at.desc()).limit(100).all()
    return jsonify({'predictions': [p.to_dict() for p in predictions]})

@auth.route('/make-admin')
@login_required
def make_admin():
    current_user.role = 'ADMIN'
    db.session.commit()
    return jsonify({'message': '관리자 변경 완료', 'role': current_user.role})


@auth.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    data = request.get_json()
    email = data.get('email', '').strip()
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': '등록된 이메일이 없어요'}), 404
    # 기존 토큰 무효화
    PasswordResetToken.query.filter_by(user_id=user.id, used=False).update({'used': True})
    db.session.commit()
    # 새 토큰 생성
    token = secrets.token_urlsafe(32)
    reset_token = PasswordResetToken(
        user_id=user.id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(hours=1)
    )
    db.session.add(reset_token)
    db.session.commit()
    # 이메일 발송
    try:
        from flask import current_app
        mail = Mail(current_app)
        reset_url = url_for('auth.reset_password', token=token, _external=True)
        msg = Message(
            subject='[Pharma Risk Analyzer] 비밀번호 재설정',
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[email]
        )
        msg.body = f'''안녕하세요, {user.username}님!

비밀번호 재설정 링크입니다:
{reset_url}

이 링크는 1시간 후 만료됩니다.
본인이 요청하지 않은 경우 이 메일을 무시하세요.

Pharma Risk Analyzer
'''
        mail.send(msg)
        return jsonify({'message': '이메일을 발송했어요! 메일함을 확인해주세요.'})
    except Exception as e:
        return jsonify({'error': f'이메일 발송 실패: {str(e)}'}), 500

@auth.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset_token = PasswordResetToken.query.filter_by(token=token, used=False).first()
    if not reset_token or reset_token.expires_at < datetime.utcnow():
        return render_template('reset_password.html', error='링크가 만료됐거나 유효하지 않아요')
    if request.method == 'GET':
        return render_template('reset_password.html', token=token)
    data = request.get_json()
    password = data.get('password', '')
    if len(password) < 6:
        return jsonify({'error': '비밀번호는 6자 이상이어야 해요'}), 400
    user = User.query.get(reset_token.user_id)
    user.password_hash = generate_password_hash(password)
    reset_token.used = True
    db.session.commit()
    return jsonify({'message': '비밀번호가 변경됐어요! 로그인해주세요.'})