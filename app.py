from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
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

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

app = Flask(__name__)
app.secret_key = '123'  # Replace with a secure key in production
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# The LLaMA 3 model will be accessed via the Ollama API, so we no longer load it here.

# Hardcoded users
USERS = {
    'admin': 'ta@2025',
    'bob': 'password2',
    'charlie': 'password3',
    'diana': 'password4',
}

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
                max_output_tokens=4000
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/process', methods=['POST'])
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

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)