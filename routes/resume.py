from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import os, uuid
from utils import extract_text_from_pdf, extract_text_from_docx, calculate_resume_match_with_openai, SAVED_FILES_FOLDER

resume_bp = Blueprint('resume', __name__)

@resume_bp.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('index.html')

@resume_bp.route('/process', methods=['POST'])
def process():
    job_description = request.form.get('job_description', '')
    resume_files = request.files.getlist('resume')
    resumes_result = []
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
                    match_result = calculate_resume_match_with_openai(job_description, resume_text)
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
        if resumes_result:
            return jsonify({'resumes': resumes_result})
    return jsonify({'resumes': []}) 