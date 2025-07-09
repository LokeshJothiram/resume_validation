from flask_sqlalchemy import SQLAlchemy
import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    # Only 'TA_TEAM' and 'admin' are valid roles
    role = db.Column(db.String(20), nullable=False)  # valid: 'TA_TEAM', 'admin'
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)
    # Add more fields as needed

class SavedAnalysis(db.Model):
    __tablename__ = 'saved_analyses'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    data = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    # Add more fields as needed

class ShortlistedResume(db.Model):
    __tablename__ = 'shortlisted_resumes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    match_percentage = db.Column(db.Float, nullable=True)
    # Add more fields as needed 

class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    username = db.Column(db.String(80), nullable=True)  # Store username for deleted users
    role = db.Column(db.String(20), nullable=True)
    action_type = db.Column(db.String(50), nullable=False)
    details = db.Column(db.JSON, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False) 

class QuestionGeneration(db.Model):
    __tablename__ = 'question_generation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    technology = db.Column(db.String(100), nullable=True)
    job_description = db.Column(db.Text, nullable=True)
    num_questions = db.Column(db.Integer, nullable=False, default=5)
    level = db.Column(db.String(20), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False) 

class SkillAssessment(db.Model):
    __tablename__ = 'skill_assessments'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    # Add more fields as needed (e.g., resume filename, result, etc.) 

class IQGFileUpload(db.Model):
    __tablename__ = 'iqg_file_uploads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False) 

class JobRequirementFile(db.Model):
    __tablename__ = 'job_requirement_files'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    job_title = db.Column(db.String(255), nullable=True) 