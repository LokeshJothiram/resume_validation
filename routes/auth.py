from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from database import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
    return render_template('login.html')

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return redirect(url_for('auth.login')) 