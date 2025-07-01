from flask import Blueprint, request, jsonify, session
from database import db, SavedAnalysis, User
import datetime

saved_bp = Blueprint('saved', __name__)

def get_current_user():
    username = session.get('user')
    if not username:
        return None
    return User.query.filter_by(username=username).first()

@saved_bp.route('/save_analysis', methods=['POST'])
def save_analysis():
    data = request.get_json()
    user = get_current_user()
    user_id = user.id if user else None
    analysis = SavedAnalysis(
        user_id=user_id,
        data=data,
        timestamp=datetime.datetime.now()
    )
    db.session.add(analysis)
    db.session.commit()
    return jsonify({'status': 'success', 'id': analysis.id})

@saved_bp.route('/list_saved_analyses', methods=['GET'])
def list_saved_analyses():
    user = get_current_user()
    if not user:
        return jsonify([])
    analyses = SavedAnalysis.query.filter_by(user_id=user.id).order_by(SavedAnalysis.timestamp.desc()).all()
    result = [
        {
            'id': a.id,
            'timestamp': a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': a.user_id,
            'data': a.data
        } for a in analyses
    ]
    return jsonify(result)

@saved_bp.route('/get_saved_analysis/<int:analysis_id>', methods=['GET'])
def get_saved_analysis(analysis_id):
    user = get_current_user()
    analysis = SavedAnalysis.query.filter_by(id=analysis_id, user_id=user.id).first_or_404()
    return jsonify(analysis.data)

@saved_bp.route('/delete_saved_analysis/<int:analysis_id>', methods=['DELETE'])
def delete_saved_analysis(analysis_id):
    user = get_current_user()
    analysis = SavedAnalysis.query.filter_by(id=analysis_id, user_id=user.id).first_or_404()
    db.session.delete(analysis)
    db.session.commit()
    return jsonify({'status': 'deleted'}) 