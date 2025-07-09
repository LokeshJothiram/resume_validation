from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash, send_from_directory
# from llama_cpp import Llama

import os
import pdfplumber
from docx import Document
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
import requests
import json
import re
import uuid
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import google.generativeai as genai
from sarvamai import SarvamAI
import tempfile
from datetime import datetime
import pytz
import glob
from routes import register_blueprints
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from database import db, User, SavedAnalysis, ShortlistedResume, ActivityLog, QuestionGeneration
from werkzeug.utils import secure_filename
from utils import log_activity, get_current_user
from sqlalchemy.exc import IntegrityError
from functools import wraps

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# Load environment variables
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DB = os.getenv('MYSQL_DB')

app = Flask(__name__)
app.secret_key = '123'  # Replace with a secure key in production
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hour session timeout
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure saved_files folder exists
SAVED_FILES_FOLDER = 'saved_files'
os.makedirs(SAVED_FILES_FOLDER, exist_ok=True)

# Ensure shortlist folder exists
SHORTLIST_FOLDER = 'shortlist'
os.makedirs(SHORTLIST_FOLDER, exist_ok=True)

# Load Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
print(GEMINI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Register blueprints
register_blueprints(app)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # Clear any existing session data
            session.clear()
            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        
        # Check if session has expired (1 hour)
        if 'login_time' in session:
            try:
                login_time = datetime.fromisoformat(session['login_time'])
                if (datetime.now() - login_time).total_seconds() > 86400:  # 24 hours
                    session.clear()
                    # Check if it's an AJAX request
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': 'Session expired'}), 401
                    return redirect(url_for('login'))
            except:
                session.clear()
                # Check if it's an AJAX request
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': 'Invalid session'}), 401
                return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

# Security headers middleware
@app.after_request
def add_security_headers(response):
    # Prevent caching of sensitive pages
    if request.endpoint and request.endpoint != 'static':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # Add other security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# The LLaMA 3 model will be accessed via the Ollama API, so we no longer load it here.

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

# Function to transcribe audio with speaker diarization using AssemblyAI
# Returns a list of candidate answers (strings)
def transcribe_audio_with_diarization(audio_path):
    ASSEMBLYAI_API_KEY = "e1e3cad33baa447aa24481cc7dbd998f"
    upload_url = "https://api.assemblyai.com/v2/upload"
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    # 1. Upload audio file
    with open(audio_path, 'rb') as f:
        response = requests.post(upload_url, headers=headers, files={"file": f})
    if response.status_code != 200:
        return f"Error uploading audio: {response.text}"
    audio_url = response.json()["upload_url"]

    # 2. Request transcription with diarization
    transcript_request = {
        "audio_url": audio_url,
        "speaker_labels": True
    }
    response = requests.post(transcript_url, json=transcript_request, headers=headers)
    if response.status_code != 200:
        return f"Error requesting transcription: {response.text}"
    transcript_id = response.json()["id"]

    # 3. Poll for completion
    while True:
        poll_response = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
        if poll_response.status_code != 200:
            return f"Error polling transcription: {poll_response.text}"
        status = poll_response.json()["status"]
        if status == "completed":
            break
        elif status == "error":
            return f"Error in transcription: {poll_response.json().get('error', 'Unknown error')}"
        import time
        time.sleep(3)

    # 4. Extract candidate's answers (speaker who talks the most)
    utterances = poll_response.json().get("utterances", [])
    if not utterances:
        return "No utterances found in transcript."
    # Count total words per speaker
    from collections import Counter
    speaker_word_counts = Counter()
    for utt in utterances:
        speaker_word_counts[utt["speaker"]] += len(utt["text"].split())
    if not speaker_word_counts:
        return "No speakers found in transcript."
    candidate_speaker = speaker_word_counts.most_common(1)[0][0]
    # Extract only candidate's answers
    answers = [utt["text"] for utt in utterances if utt["speaker"] == candidate_speaker]
    return answers

# Function to evaluate resume match using Gemini 1.5 Flash
# Ollama code is commented out below
def calculate_resume_match_with_gemini(job_description, resume_text):
    prompt = f"""
    You are an expert technical recruiter. Analyze the following job description and resume, and determine how well the resume matches the job description. Provide a match percentage and a brief explanation for your reasoning.

    Job Description: {job_description}

    Resume: {resume_text}

    Output ONLY the following JSON object:
    {{
      \"match_percentage\": <percentage>,
      \"explanation\": \"<explanation>\"
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000
            )
        )
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {"match_percentage": 0, "explanation": "Could not parse Gemini output."}
    except Exception as e:
        return {"match_percentage": 0, "explanation": "Service temporarily unavailable. Please try again later or contact support. (Error code: GEMINI-001)"}

# Function to evaluate technical proficiency using Gemini 1.5 Flash
# Ollama code is commented out below
def evaluate_technical_proficiency_with_gemini(transcription, technology, tech_questions=None):
    if tech_questions:
        questions = [q.strip() for q in tech_questions.split('\n') if q.strip()]
        questions_json = json.dumps(questions)
        prompt = f"""
You are an expert technical interviewer. You are given a list of technology interview questions for a {technology} role, and the candidate's transcribed answers from an interview.

For each question in the list below, do the following:
- Find the candidate's answer (if any) from the transcript.
- Grade the answer out of 10 (0-10).
- Provide a brief explanation for the grade.

Return your response as a single JSON object with:
- "question_grades": [an array where each element is an object with "question", "answer", "score", "explanation"]
- "technical_score": <score>
- "technical_explanation": "<explanation> (must be at least 50 words)"
- "depth_score": <score>
- "depth_explanation": "<explanation>"
- "relevance_score": <score>
- "relevance_explanation": "<explanation>"
- "communication_score": <score>
- "communication_explanation": "<explanation>"
- "clarity_score": <score>
- "clarity_explanation": "<explanation>"
- "confidence_score": <score>
- "confidence_explanation": "<explanation>"
- "problem_solving_score": <score>
- "problem_solving_explanation": "<explanation>"

The technical explanation must be at least 50 words.

Questions:
{questions_json}

Transcript:
{transcription}
"""
    else:
        prompt = f"""
You are an expert technical interviewer. Analyze the following transcribed interview answers for a {technology} role and evaluate the candidate on multiple dimensions:

1. Technical proficiency (score out of 10, with explanation)
2. Depth of technical knowledge (score out of 10, with explanation)
3. Relevance of answers to technology (score out of 10, with explanation)
4. Communication skills (score out of 10, with explanation)
5. Clarity of explanation (score out of 10, with explanation)
6. Confidence (score out of 10, with explanation)
7. Problem-solving approach (score out of 10, with explanation)

Provide your analysis in the following JSON format:
{{
  "technical_score": <score>,
  "technical_explanation": "<explanation> (must be at least 50 words)",
  "depth_score": <score>,
  "depth_explanation": "<explanation>",
  "relevance_score": <score>,
  "relevance_explanation": "<explanation>",
  "communication_score": <score>,
  "communication_explanation": "<explanation>",
  "clarity_score": <score>,
  "clarity_explanation": "<explanation>",
  "confidence_score": <score>,
  "confidence_explanation": "<explanation>",
  "problem_solving_score": <score>,
  "problem_solving_explanation": "<explanation>"
}}

The technical explanation must be at least 50 words.

Answers:
{transcription}
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000
            )
        )
        match_obj = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match_obj:
            try:
                result = json.loads(match_obj.group(0))
                return result
            except Exception:
                pass
        return {
            "technical_score": 0,
            "technical_explanation": "Could not parse Gemini output.",
            "depth_score": 0,
            "depth_explanation": "",
            "relevance_score": 0,
            "relevance_explanation": "",
            "communication_score": 0,
            "communication_explanation": "",
            "clarity_score": 0,
            "clarity_explanation": "",
            "confidence_score": 0,
            "confidence_explanation": "",
            "problem_solving_score": 0,
            "problem_solving_explanation": "",
            "question_grades": []
        }
    except Exception as e:
        return {
            "technical_score": 0,
            "technical_explanation": "Service temporarily unavailable. Please try again later or contact support. (Error code: GEMINI-002)",
            "depth_score": 0,
            "depth_explanation": "",
            "relevance_score": 0,
            "relevance_explanation": "",
            "communication_score": 0,
            "communication_explanation": "",
            "clarity_score": 0,
            "clarity_explanation": "",
            "confidence_score": 0,
            "confidence_explanation": "",
            "problem_solving_score": 0,
            "problem_solving_explanation": "",
            "question_grades": []
        }

def split_audio(audio_path, chunk_duration_ms=29000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            chunk.export(temp_file.name, format="wav")
            chunks.append(temp_file.name)
    return chunks

# --- Sarvam Speech-to-Text Integration using SDK with chunking ---
def transcribe_audio_with_sarvam(audio_path, model="saarika:v2.5", language_code="en-IN"):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        if duration_seconds <= 30:
            with open(audio_path, "rb") as audio_file:
                response = client.speech_to_text.transcribe(
                    file=audio_file,
                    model=model,
                    language_code=language_code
                )
                return response.transcript
        else:
            chunks = split_audio(audio_path)
            transcripts = []
            for chunk_path in chunks:
                with open(chunk_path, "rb") as audio_file:
                    response = client.speech_to_text.transcribe(
                        file=audio_file,
                        model=model,
                        language_code=language_code
                    )
                    transcripts.append(response.transcript)
            # Clean up temp files
            for chunk_path in chunks:
                try:
                    os.unlink(chunk_path)
                except:
                    pass
            return ' '.join(transcripts)
    except Exception as e:
        return f"Error contacting Sarvam API: {e}"

# --- Commented out Ollama code ---
# def calculate_resume_match_with_ollama(job_description, resume_text):
#     prompt = f"""
#     You are an expert technical recruiter. Analyze the following job description and resume, and determine how well the resume matches the job description. Provide a match percentage and a brief explanation for your reasoning.
# 
#     Job Description: {job_description}
# 
#     Resume: {resume_text}
# 
#     Output ONLY the following JSON object:
#     {{
#       "match_percentage": <percentage>,
#       "explanation": "<explanation>"
#     }}
#     """
#     
#     ollama_api_url = "http://localhost:11434/api/generate"
#     payload = {
#         # "model": "gemma:2b",
#         # "model": "tinyllama",
#         "model": "llama3-gguf",
#         "prompt": prompt,
#         "format": "json",
#         "stream": False
#     }
# 
#     try:
#         response = requests.post(ollama_api_url, json=payload)
#         response.raise_for_status() # Raise an exception for bad status codes
#         
#         response_data = response.json()
#         result = json.loads(response_data['response'])
#         return result
#     except requests.exceptions.RequestException as e:
#         return {"match_percentage": 0, "explanation": f"Error contacting Ollama API: {e}"}
#     except (KeyError, json.JSONDecodeError):
#         return {"match_percentage": 0, "explanation": "Error parsing model output from Ollama."}
#
# def evaluate_technical_proficiency(transcription, technology):
#     prompt = f"""
#     You are an expert technical interviewer. Analyze the following transcribed interview response for a {technology} role and evaluate the candidate's technical proficiency. Provide a score out of 10 based on technical accuracy, depth, and relevance to {technology}. Also, provide a brief explanation.
# 
#     Transcription: {transcription}
# 
#     Output ONLY the following JSON object:
#     {{
#       "score": <score>,
#       "explanation": "<explanation>"
#     }}
#     """
#     
#     ollama_api_url = "http://localhost:11434/api/generate"
#     payload = {
#         # "model": "gemma:2b",
#         "model": "llama3-gguf",
#         # "model": "mistral",
#         "prompt": prompt,
#         "format": "json",
#         "stream": False
#     }
# 
#     try:
#         response = requests.post(ollama_api_url, json=payload)
#         response.raise_for_status() # Raise an exception for bad status codes
#         
#         # The response from Ollama is a JSON string in the 'response' field
#         response_data = response.json()
#         result = json.loads(response_data['response'])
#         return result
#     except requests.exceptions.RequestException as e:
#         return {"score": 0, "explanation": f"Error contacting Ollama API: {e}"}
#     except (KeyError, json.JSONDecodeError):
#         return {"score": 0, "explanation": "Error parsing model output from Ollama."}

def separate_hr_candidate_with_gemini(transcript):
    """
    The following is a transcript of a job interview between an HR interviewer and a candidate.
    Please separate the transcript into a conversation history, labeling each line as either 'HR:' or 'Candidate:'.
    Do not use a table or columns. Just alternate lines starting with 'HR:' or 'Candidate:' as appropriate.
    """
    prompt = f"""
    The following is a transcript of a job interview between an HR interviewer and a candidate.
    Please separate the transcript into a conversation history, labeling each line as either 'HR:' or 'Candidate:'.
    Do not use a table or columns. Just alternate lines starting with 'HR:' or 'Candidate:' as appropriate.

    Transcript:
    {transcript}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000
            )
        )
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

# Helper function to get current IST time
def get_ist_now():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Always clear session when visiting login page
    session.clear()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session.permanent = True
            session['user'] = username
            session['user_id'] = user.id
            session['role'] = user.role
            session['login_time'] = datetime.now().isoformat()
            log_activity(user, 'login')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    print("LOGOUT ROUTE CALLED")
    user = get_current_user()
    print("Logging out user:", user.username if user else None)
    log_activity(user, 'logout', {'message': 'User logged out', 'username': user.username if user else None})
    session.clear()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    username = session.get('user')
    return render_template('index.html', username=username)

# Handle 404 errors
@app.errorhandler(404)
def not_found_error(error):
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('404.html'), 404

# Handle 401 errors
@app.errorhandler(401)
def unauthorized_error(error):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'error': 'Unauthorized'}), 401
    return redirect(url_for('login'))

# Handle 403 errors
@app.errorhandler(403)
def forbidden_error(error):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'error': 'Forbidden'}), 403
    return redirect(url_for('login'))

@app.route('/process', methods=['POST'])
@login_required
def process():
    job_description = request.form.get('job_description', '')
    resume_files = request.files.getlist('resume')
    audio_file = request.files.get('audio')
    technology = request.form.get('technology', 'General')
    tech_questions = request.form.get('tech_questions', '').strip()

    print(f"Selected technology: {technology}")
    if audio_file and audio_file.filename:
        print(f"Uploaded audio file: {audio_file.filename}")
    if tech_questions:
        print(f"Uploaded technology questions: {tech_questions[:60]}...")

    response = {}

    # Process multiple resumes
    resumes_result = []
    if resume_files and job_description.strip():
        for resume_file in resume_files:
            if resume_file and resume_file.filename:
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + "_" + resume_file.filename)
                resume_file.save(resume_path)
                resume_text = ""
                if resume_file.filename.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(resume_path)
                elif resume_file.filename.endswith('.docx'):
                    resume_text = extract_text_from_docx(resume_path)
                else:
                    resumes_result.append({
                        'filename': resume_file.filename,
                        'error': 'Unsupported file type.'
                    })
                    os.remove(resume_path)
                    continue
                if "Error" not in resume_text:
                    match_result = calculate_resume_match_with_gemini(job_description, resume_text)
                    resumes_result.append({
                        'filename': resume_file.filename,
                        'match_percentage': match_result.get('match_percentage'),
                        'explanation': match_result.get('explanation')
                    })
                else:
                    resumes_result.append({
                        'filename': resume_file.filename,
                        'error': resume_text
                    })
                os.remove(resume_path)
        if resumes_result:
            response['resumes'] = resumes_result

    # Process audio (Sarvam API)
    if audio_file and audio_file.filename:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + "_" + audio_file.filename)
        audio_file.save(audio_path)
        transcript = transcribe_audio_with_sarvam(audio_path)
        if isinstance(transcript, str) and transcript.startswith("Error"):
            response['audio_error'] = transcript
        elif isinstance(transcript, str) and transcript.strip():
            response['transcription'] = transcript
            # Separate HR and candidate messages using Gemini
            separated_dialog = separate_hr_candidate_with_gemini(transcript)
            response['separated_dialog'] = separated_dialog
            # Extract only candidate's messages
            candidate_lines = []
            for line in separated_dialog.splitlines():
                if line.strip().startswith('Candidate:'):
                    candidate_lines.append(line.replace('Candidate:', '').strip())
            candidate_text = '\n'.join(candidate_lines)
            # Run technical proficiency evaluation on only candidate's answers
            tech_eval = evaluate_technical_proficiency_with_gemini(
                f"The following are only the candidate's answers from an interview transcript:\n{candidate_text}",
                technology,
                tech_questions=tech_questions
            )
            response['technical_score'] = tech_eval.get('technical_score', 0)
            response['technical_explanation'] = tech_eval.get('technical_explanation', '')
            response['depth_score'] = tech_eval.get('depth_score', 0)
            response['depth_explanation'] = tech_eval.get('depth_explanation', '')
            response['relevance_score'] = tech_eval.get('relevance_score', 0)
            response['relevance_explanation'] = tech_eval.get('relevance_explanation', '')
            response['communication_score'] = tech_eval.get('communication_score', 0)
            response['communication_explanation'] = tech_eval.get('communication_explanation', '')
            response['clarity_score'] = tech_eval.get('clarity_score', 0)
            response['clarity_explanation'] = tech_eval.get('clarity_explanation', '')
            response['confidence_score'] = tech_eval.get('confidence_score', 0)
            response['confidence_explanation'] = tech_eval.get('confidence_explanation', '')
            response['problem_solving_score'] = tech_eval.get('problem_solving_score', 0)
            response['problem_solving_explanation'] = tech_eval.get('problem_solving_explanation', '')
            response['question_grades'] = tech_eval.get('question_grades', [])
        else:
            response['audio_error'] = "No transcript detected."
        os.remove(audio_path)

    # After processing audio (if any was analyzed), log a SkillAssessment event
    from database import SkillAssessment, db
    user = get_current_user()
    if user and audio_file and audio_file.filename and 'transcription' in response:
        assessment = SkillAssessment(user_id=user.id)
        db.session.add(assessment)
        db.session.commit()

    return jsonify(response)

# Save analysis endpoint
@app.route('/save_analysis', methods=['POST'])
@login_required
def save_analysis():
    data = request.get_json()
    timestamp = get_ist_now().strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_{timestamp}.json"
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log_activity(get_current_user(), 'save_analysis', {'filename': filename})
    return jsonify({'status': 'success', 'filename': filename})

# List all saved analysis files
@app.route('/list_saved_analyses', methods=['GET'])
@login_required
def list_saved_analyses():
    files = glob.glob(os.path.join(SAVED_FILES_FOLDER, 'analysis_*.json'))
    files.sort(reverse=True)
    result = []
    for f in files:
        fname = os.path.basename(f)
        # Extract timestamp from filename
        ts = fname.replace('analysis_', '').replace('.json', '')
        result.append({'filename': fname, 'timestamp': ts})
    return jsonify(result)

# Get a saved analysis file's contents
@app.route('/get_saved_analysis/<filename>', methods=['GET'])
@login_required
def get_saved_analysis(filename):
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

# Delete a saved analysis file
@app.route('/delete_saved_analysis/<filename>', methods=['DELETE'])
@login_required
def delete_saved_analysis(filename):
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    os.remove(filepath)
    log_activity(get_current_user(), 'delete_saved_analysis', {'filename': filename})
    return jsonify({'status': 'deleted'})

@app.route('/shortlist_resume', methods=['POST'])
@login_required
def shortlist_resume():
    resume_file = request.files.get('shortlist_resume')
    timestamp = request.form.get('timestamp', '')
    match_percentage = request.form.get('match_percentage', None)
    user = get_current_user()
    user_id = user.id if user else None
    if resume_file and resume_file.filename:
        from database import ShortlistedResume, db
        import uuid
        from datetime import datetime
        original_filename = secure_filename(resume_file.filename)
        file_data = resume_file.read()
        unique_id = uuid.uuid4().hex[:8]
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        if '.' in original_filename:
            name, ext = original_filename.rsplit('.', 1)
            unique_filename = f"{unique_id}_{name}_{timestamp_str}.{ext}"
        else:
            unique_filename = f"{unique_id}_{original_filename}_{timestamp_str}"
        shortlist = ShortlistedResume(
            user_id=user_id,
            original_filename=unique_filename,
            file_data=file_data,
            timestamp=datetime.now(),
            match_percentage=match_percentage
        )
        db.session.add(shortlist)
        db.session.commit()
        log_activity(user, 'shortlist_resume', {'filename': unique_filename})
        return jsonify({'status': 'success', 'id': shortlist.id})
    return jsonify({'status': 'error', 'message': 'No file uploaded'})

@app.route('/list_shortlisted', methods=['GET'])
@login_required
def list_shortlisted():
    user = get_current_user()
    if not user:
        return jsonify([])
    from database import ShortlistedResume
    resumes = ShortlistedResume.query.filter_by(user_id=user.id).order_by(ShortlistedResume.timestamp.desc()).all()
    result = [
        {
            'id': r.id,
            'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'original_filename': r.original_filename,
            'user_id': r.user_id,
            'match_percentage': r.match_percentage
        } for r in resumes
    ]
    return jsonify(result)

@app.route('/get_shortlisted/<int:resume_id>', methods=['GET'])
@login_required
def get_shortlisted(resume_id):
    user = get_current_user()
    from database import ShortlistedResume
    r = ShortlistedResume.query.filter_by(id=resume_id, user_id=user.id).first_or_404()
    return jsonify({
        'id': r.id,
        'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'original_filename': r.original_filename,
        'user_id': r.user_id,
        'match_percentage': r.match_percentage
    })

@app.route('/delete_shortlisted/<int:resume_id>', methods=['DELETE'])
@login_required
def delete_shortlisted(resume_id):
    user = get_current_user()
    from database import ShortlistedResume, db
    r = ShortlistedResume.query.filter_by(id=resume_id, user_id=user.id).first_or_404()
    db.session.delete(r)
    db.session.commit()
    log_activity(user, 'delete_shortlisted', {'resume_id': resume_id})
    return jsonify({'status': 'deleted'})

@app.route('/shortlist/<int:resume_id>')
@login_required
def download_shortlisted_resume(resume_id):
    from database import ShortlistedResume
    from flask import send_file
    import io
    r = ShortlistedResume.query.get_or_404(resume_id)
    return send_file(
        io.BytesIO(r.file_data),
        as_attachment=True,
        download_name=r.original_filename
    )

@app.route('/admin_analytics', methods=['GET'])
@login_required
def admin_analytics():
    # Query all users and their roles
    users = User.query.all()
    role_stats = {}
    for user in users:
        role = user.role
        if role not in role_stats:
            role_stats[role] = {'shortlisted': 0, 'analyses': 0, 'questions': 0, 'users': 0}
        role_stats[role]['users'] += 1
    # Count shortlisted resumes per user
    for sr in ShortlistedResume.query.all():
        user = db.session.get(User, sr.user_id) if sr.user_id else None
        if user:
            role_stats[user.role]['shortlisted'] += 1
    # Count saved analyses per user
    for sa in SavedAnalysis.query.all():
        user = db.session.get(User, sa.user_id) if sa.user_id else None
        if user:
            role_stats[user.role]['analyses'] += 1
    # Count questions generated per user (assuming you have a QuestionGeneration model)
    try:
        from database import QuestionGeneration
        for qg in QuestionGeneration.query.all():
            user = db.session.get(User, qg.user_id) if qg.user_id else None
            if user:
                role_stats[user.role]['questions'] += 1
    except ImportError:
        pass  # If you don't have a QuestionGeneration model, skip
    # Format for frontend
    result = []
    for role, stats in role_stats.items():
        result.append({
            'role': role,
            'users': stats['users'],
            'shortlisted': stats['shortlisted'],
            'analyses': stats['analyses'],
            'questions': stats['questions']
        })
    return jsonify(result)

@app.route('/admin_analytics_users', methods=['GET'])
@login_required
def admin_analytics_users():
    users = User.query.all()
    shortlisted_counts = {u.id: 0 for u in users}
    analyses_counts = {u.id: 0 for u in users}
    for sr in ShortlistedResume.query.all():
        if sr.user_id:
            shortlisted_counts[sr.user_id] = shortlisted_counts.get(sr.user_id, 0) + 1
    for sa in SavedAnalysis.query.all():
        if sa.user_id:
            analyses_counts[sa.user_id] = analyses_counts.get(sa.user_id, 0) + 1
    result = []
    for u in users:
        result.append({
            'username': u.username,
            'role': u.role,
            'shortlisted': shortlisted_counts.get(u.id, 0),
            'analyses': analyses_counts.get(u.id, 0)
        })
    return jsonify(result)

@app.route('/activity_logs', methods=['GET'])
@login_required
def get_activity_logs():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    user_id = request.args.get('user_id')
    role = request.args.get('role')
    action_type = request.args.get('action_type')
    limit = int(request.args.get('limit', 100))
    q = ActivityLog.query
    if user_id:
        q = q.filter_by(user_id=user_id)
    if role:
        q = q.filter_by(role=role)
    if action_type:
        q = q.filter_by(action_type=action_type)
    logs = q.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
    return jsonify([
        {
            'id': log.id,
            'user_id': log.user_id,
            'username': log.username,
            'role': log.role,
            'action_type': log.action_type,
            'details': log.details,
            'timestamp': log.timestamp.isoformat()
        }
        for log in logs
    ])

@app.route('/user_activity_logs', methods=['GET'])
@login_required
def get_user_activity_logs():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 403
    limit = int(request.args.get('limit', 10))
    q = ActivityLog.query.filter_by(user_id=user.id)
    logs = q.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
    return jsonify([
        {
            'id': log.id,
            'user_id': log.user_id,
            'username': log.username,
            'role': log.role,
            'action_type': log.action_type,
            'details': log.details,
            'timestamp': log.timestamp.isoformat()
        }
        for log in logs
    ])

@app.route('/admin_users', methods=['GET'])
@login_required
def admin_list_users():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    users = User.query.all()
    return jsonify([
        {'id': u.id, 'username': u.username, 'role': u.role, 'created_at': u.created_at.isoformat(), 'active': True} for u in users
    ])

@app.route('/admin_users', methods=['POST'])
@login_required
def admin_add_user():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.json
    if not data.get('username') or not data.get('password') or not data.get('role'):
        return jsonify({'error': 'Missing fields'}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username exists'}), 400
    user = User(username=data['username'], password=data['password'], role=data['role'])
    db.session.add(user)
    db.session.commit()
    log_activity(get_current_user(), 'admin_add_user', {
        'added_user': user.username,
        'added_user_id': user.id,
        'added_user_role': user.role
    })
    return jsonify({'status': 'success', 'id': user.id})

@app.route('/admin_users/<int:user_id>', methods=['PUT'])
@login_required
def admin_edit_user(user_id):
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    data = request.json
    if 'username' in data:
        user.username = data['username']
    if 'role' in data:
        user.role = data['role']
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'status': 'error', 'error': 'Username already exists.'}), 400
    log_activity(get_current_user(), 'admin_edit_user', {
        'edited_user': user.username,
        'edited_user_id': user.id,
        'edited_user_role': user.role
    })
    return jsonify({'status': 'success'})

@app.route('/admin_users/<int:user_id>', methods=['DELETE'])
@login_required
def admin_delete_user(user_id):
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    # Log the activity BEFORE deleting the user and their logs
    log_activity(get_current_user(), 'admin_delete_user', {'deleted_user': user.username, 'deleted_user_id': user.id})
    # Delete all related records
    SavedAnalysis.query.filter_by(user_id=user.id).delete()
    ShortlistedResume.query.filter_by(user_id=user.id).delete()
    ActivityLog.query.filter_by(user_id=user.id).delete()
    QuestionGeneration.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/admin_users/reset_password', methods=['POST'])
@login_required
def admin_reset_password():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.json
    user = User.query.get(data.get('user_id'))
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.password = data.get('new_password')
    db.session.commit()
    log_activity(get_current_user(), 'admin_reset_password', {
        'reset_user': user.username,
        'reset_user_id': user.id
    })
    return jsonify({'status': 'success'})

@app.route('/admin_users/import', methods=['POST'])
@login_required
def admin_import_users():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    users = request.json.get('users', [])
    added = 0
    for u in users:
        if not u.get('username') or not u.get('password') or not u.get('role'):
            continue
        if User.query.filter_by(username=u['username']).first():
            continue
        user = User(username=u['username'], password=u['password'], role=u['role'])
        db.session.add(user)
        db.session.commit()
        log_activity(get_current_user(), 'admin_import_user', {
            'imported_user': user.username,
            'imported_user_id': user.id,
            'imported_user_role': user.role
        })
        added += 1
    return jsonify({'status': 'success', 'added': added})

@app.route('/admin_users/export', methods=['GET'])
@login_required
def admin_export_users():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    users = User.query.all()
    data = [
        {'id': u.id, 'username': u.username, 'role': u.role, 'created_at': u.created_at.isoformat()} for u in users
    ]
    return jsonify({'users': data})

@app.route('/user_dashboard_stats', methods=['GET'])
@login_required
def user_dashboard_stats():
    # This route is also used for session validation
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' and request.args.get('validate_session'):
        return jsonify({'status': 'valid'})
    
    user = get_current_user()
    if not user:
        return jsonify({
            'resumes_analyzed': 0,
            'resumes_analyzed_delta': '',
            'skills_assessed': 0,
            'skills_assessed_delta': '',
            'questions_generated': 0,
            'questions_generated_delta': '',
            'shortlisted_candidates': 0,
            'shortlisted_candidates_delta': ''
        })
    from database import ShortlistedResume, SavedAnalysis, SkillAssessment
    # Resumes analyzed = SavedAnalysis by user
    from datetime import datetime, timedelta
    now = datetime.now()
    last_month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    # This month
    resumes_analyzed = SavedAnalysis.query.filter_by(user_id=user.id).count()
    # Last month
    resumes_analyzed_last_month = SavedAnalysis.query.filter(
        SavedAnalysis.user_id==user.id,
        SavedAnalysis.timestamp >= last_month,
        SavedAnalysis.timestamp < now.replace(day=1)
    ).count()
    resumes_analyzed_delta = ''
    if resumes_analyzed_last_month > 0:
        delta = resumes_analyzed - resumes_analyzed_last_month
        percent = int((delta / resumes_analyzed_last_month) * 100)
        resumes_analyzed_delta = f"{percent:+d}% from last month"
    # Skills assessed = SkillAssessment by user
    skills_assessed = SkillAssessment.query.filter_by(user_id=user.id).count()
    skills_assessed_last_month = SkillAssessment.query.filter(
        SkillAssessment.user_id==user.id,
        SkillAssessment.timestamp >= last_month,
        SkillAssessment.timestamp < now.replace(day=1)
    ).count()
    skills_assessed_delta = ''
    if skills_assessed_last_month > 0:
        delta = skills_assessed - skills_assessed_last_month
        percent = int((delta / skills_assessed_last_month) * 100)
        skills_assessed_delta = f"{percent:+d}% from last month"
    # Interview questions generated (if model exists)
    try:
        from database import QuestionGeneration
        questions_generated = QuestionGeneration.query.filter_by(user_id=user.id).count()
        questions_generated_last_month = QuestionGeneration.query.filter(
            QuestionGeneration.user_id==user.id,
            QuestionGeneration.timestamp >= last_month,
            QuestionGeneration.timestamp < now.replace(day=1)
        ).count()
        questions_generated_delta = ''
        if questions_generated_last_month > 0:
            delta = questions_generated - questions_generated_last_month
            percent = int((delta / questions_generated_last_month) * 100)
            questions_generated_delta = f"{percent:+d}% from last month"
    except ImportError:
        questions_generated = 0
        questions_generated_delta = ''
    # Shortlisted candidates
    shortlisted_candidates = ShortlistedResume.query.filter_by(user_id=user.id).count()
    shortlisted_candidates_last_month = ShortlistedResume.query.filter(
        ShortlistedResume.user_id==user.id,
        ShortlistedResume.timestamp >= last_month,
        ShortlistedResume.timestamp < now.replace(day=1)
    ).count()
    shortlisted_candidates_delta = ''
    if shortlisted_candidates_last_month > 0:
        delta = shortlisted_candidates - shortlisted_candidates_last_month
        percent = int((delta / shortlisted_candidates_last_month) * 100)
        shortlisted_candidates_delta = f"{percent:+d}% from last month"
    return jsonify({
        'resumes_analyzed': resumes_analyzed,
        'resumes_analyzed_delta': resumes_analyzed_delta,
        'skills_assessed': skills_assessed,
        'skills_assessed_delta': skills_assessed_delta,
        'questions_generated': questions_generated,
        'questions_generated_delta': questions_generated_delta,
        'shortlisted_candidates': shortlisted_candidates,
        'shortlisted_candidates_delta': shortlisted_candidates_delta
    })

# Job Requirement Upload Routes
@app.route('/admin/upload-job-requirement', methods=['POST'])
@login_required
def admin_upload_job_requirement():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from database import JobRequirementFile, db
        from utils import get_current_user
        uploaded_file = request.files.get('job_file')
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({'error': 'File is required'}), 400
        original_filename = secure_filename(uploaded_file.filename)
        file_data = uploaded_file.read()
        from datetime import datetime
        user = get_current_user()
        user_id = user.id if user else None
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        if '.' in original_filename:
            name, ext = original_filename.rsplit('.', 1)
            unique_filename = f"{unique_id}_{name}_{timestamp_str}.{ext}"
        else:
            unique_filename = f"{unique_id}_{original_filename}_{timestamp_str}"
        job_title = unique_filename.rsplit('.', 1)[0]
        job_file = JobRequirementFile(
            user_id=user_id,
            original_filename=unique_filename,
            file_data=file_data,
            timestamp=datetime.now(),
            job_title=job_title
        )
        db.session.add(job_file)
        db.session.commit()
        log_activity(user, 'admin_upload_job_requirement', {
            'filename': original_filename,
            'has_file': True
        })
        return jsonify({'status': 'success', 'message': 'Job requirement uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/job-requirements', methods=['GET'])
@login_required
def admin_get_job_requirements():
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from database import JobRequirementFile
        files = []
        for f in JobRequirementFile.query.order_by(JobRequirementFile.timestamp.desc()).all():
            files.append({
                'id': f.id,
                'filename': f.original_filename,
                'job_title': f.job_title,
                'upload_date': f.timestamp.strftime('%Y-%m-%d %H:%M'),
                'size': len(f.file_data)
            })
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/download-job-requirement/<int:file_id>', methods=['GET'])
@login_required
def admin_download_job_requirement(file_id):
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from database import JobRequirementFile
        from flask import send_file
        import io
        file_record = JobRequirementFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        return send_file(
            io.BytesIO(file_record.file_data),
            as_attachment=True,
            download_name=file_record.original_filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/delete-job-requirement/<int:file_id>', methods=['DELETE'])
@login_required
def admin_delete_job_requirement(file_id):
    if 'role' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from database import JobRequirementFile, db
        file_record = JobRequirementFile.query.get(file_id)
        if not file_record:
            return jsonify({'error': 'File not found'}), 404
        db.session.delete(file_record)
        db.session.commit()
        log_activity(get_current_user(), 'admin_delete_job_requirement', {
            'deleted_file_id': file_id
        })
        return jsonify({'status': 'success', 'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job-requirements', methods=['GET'])
@login_required
def get_job_requirements_public():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from database import JobRequirementFile
        files = []
        for f in JobRequirementFile.query.order_by(JobRequirementFile.timestamp.desc()).all():
            files.append({
                'id': f.id,
                'filename': f.original_filename,
                'job_title': f.job_title,
                'upload_date': f.timestamp.strftime('%Y-%m-%d %H:%M'),
                'size': len(f.file_data)
            })
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-job-requirement')
def test_job_requirement():
    return send_from_directory('.', 'test_job_requirement.html')

@app.route('/upload_iqg_file', methods=['POST'])
@login_required
def upload_iqg_file():
    from database import IQGFileUpload, db
    from utils import get_current_user
    uploaded_file = request.files.get('iqg_file')
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    original_filename = secure_filename(uploaded_file.filename)
    file_data = uploaded_file.read()
    from datetime import datetime
    user = get_current_user()
    user_id = user.id if user else None
    iqg_file = IQGFileUpload(
        user_id=user_id,
        original_filename=original_filename,
        file_data=file_data,
        timestamp=datetime.now()
    )
    db.session.add(iqg_file)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'File uploaded successfully'})

@app.route('/list_iqg_files', methods=['GET'])
@login_required
def list_iqg_files():
    from database import IQGFileUpload
    files = []
    for f in IQGFileUpload.query.order_by(IQGFileUpload.timestamp.desc()).all():
        files.append({
            'id': f.id,
            'original_filename': f.original_filename,
            'upload_date': f.timestamp.strftime('%Y-%m-%d %H:%M'),
            'user_id': f.user_id
        })
    return jsonify({'files': files})

@app.route('/download_iqg_file/<int:file_id>')
@login_required
def download_iqg_file(file_id):
    from database import IQGFileUpload
    from flask import send_file
    import io
    file_record = IQGFileUpload.query.get(file_id)
    if not file_record:
        return jsonify({'error': 'File not found'}), 404
    return send_file(
        io.BytesIO(file_record.file_data),
        as_attachment=True,
        download_name=file_record.original_filename
    )

@app.route('/delete_iqg_file/<int:file_id>', methods=['DELETE'])
@login_required
def delete_iqg_file(file_id):
    from database import IQGFileUpload, db
    file = IQGFileUpload.query.get(file_id)
    if not file:
        return {'status': 'error', 'message': 'File not found'}, 404
    db.session.delete(file)
    db.session.commit()
    return {'status': 'success', 'message': 'File deleted'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0',port=5010,debug=True)