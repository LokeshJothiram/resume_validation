from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import os, uuid
from utils import extract_text_from_pdf, extract_text_from_docx, calculate_resume_match_with_gemini, SAVED_FILES_FOLDER
from functools import wraps

resume_bp = Blueprint('resume', __name__)

# Authentication decorator for blueprint
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # Clear any existing session data
            session.clear()
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@resume_bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    username = session.get('user')
    return render_template('index.html', username=username)

@resume_bp.route('/process', methods=['POST'])
@login_required
def process():
    from database import SkillAssessment, db, JobRequirementFile
    from utils import get_current_user
    import tempfile
    user = get_current_user()
    if user:
        assessment = SkillAssessment(user_id=user.id)
        db.session.add(assessment)
        db.session.commit()
        from utils import log_activity
        log_activity(user, 'skills_assessed', {'assessment_id': assessment.id})
    job_description = request.form.get('job_description', '')
    resume_files = request.files.getlist('resume')
    resumes_result = []
    # NEW: Handle job requirement file analysis from DB
    job_requirement_id = request.form.get('job_requirement_id', '')
    job_requirement_analysis = None
    if job_requirement_id:
        job_req_record = JobRequirementFile.query.filter_by(original_filename=job_requirement_id).first()
        if job_req_record:
            file_data = job_req_record.file_data
            filename = job_req_record.original_filename
            job_req_text = ''
            if filename.endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(file_data)
                    tmp.flush()
                    job_req_text = extract_text_from_pdf(tmp.name)
            elif filename.endswith('.docx'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(file_data)
                    tmp.flush()
                    job_req_text = extract_text_from_docx(tmp.name)
            elif filename.endswith('.txt'):
                job_req_text = file_data.decode('utf-8')
            else:
                job_req_text = f'Unsupported file type: {filename}'
            if 'Error' not in job_req_text:
                job_requirement_analysis = calculate_resume_match_with_gemini(job_req_text, job_req_text)
                job_requirement_analysis['filename'] = filename
            else:
                job_requirement_analysis = {'filename': filename, 'error': job_req_text}
        else:
            job_requirement_analysis = {'filename': job_requirement_id, 'error': 'File not found in database.'}
    # Resume analysis (existing)
    if resume_files and job_description.strip():
        for resume_file in resume_files:
            if resume_file and resume_file.filename:
                resume_path = os.path.join('uploads', str(uuid.uuid4()) + '_' + resume_file.filename)
                resume_file.save(resume_path)
                resume_text = ''
                if resume_file.filename.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(resume_path)
                elif resume_file.filename.endswith('.docx'):
                    resume_text = extract_text_from_docx(resume_path)
                else:
                    resumes_result.append({'filename': resume_file.filename, 'error': 'Unsupported file type.'})
                    os.remove(resume_path)
                    continue
                if 'Error' not in resume_text:
                    match_result = calculate_resume_match_with_gemini(job_description, resume_text)
                    resumes_result.append({
                        'filename': resume_file.filename,
                        'match_percentage': match_result.get('match_percentage'),
                        'explanation': match_result.get('explanation'),
                        'skills_matched': match_result.get('skills_matched', 0),
                        'total_skills': match_result.get('total_skills', 0),
                        'experience_match': match_result.get('experience_match', 0),
                        'education_match': match_result.get('education_match', 0),
                        'certifications_match': match_result.get('certifications_match', 0),
                        'role_match': match_result.get('role_match', 0)
                    })
                else:
                    resumes_result.append({'filename': resume_file.filename, 'error': resume_text})
                os.remove(resume_path)
    
    # Prepare response
    response = {'resumes': resumes_result}
    if job_requirement_analysis:
        response['job_requirement_analysis'] = job_requirement_analysis
    
    return jsonify(response) 