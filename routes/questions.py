from flask import Blueprint, request, jsonify, session
import re
import google.generativeai as genai
from database import db, QuestionGeneration
from utils import get_current_user
from functools import wraps

questions_bp = Blueprint('questions', __name__)

# Authentication decorator for blueprint
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            # Clear any existing session data
            session.clear()
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@questions_bp.route('/generate_questions', methods=['POST'])
@login_required
def generate_questions():
    data = request.get_json()
    technology = data.get('technology', '').strip()
    job_description = data.get('job_description', '').strip()
    num_questions = int(data.get('num_questions', 5))
    level = data.get('level', 'easy').strip().lower()
    if num_questions > 20:
        num_questions = 20
    if num_questions < 1:
        num_questions = 1
    prompt = f"""
You are an expert interviewer. Generate a list of {num_questions} relevant, challenging, and up-to-date interview questions for a {technology or 'General'} role.
"""
    if level in ['easy', 'medium', 'hard']:
        prompt += f"\nThe questions should be at a {level} level of difficulty."
    if job_description:
        prompt += f"\nThe job description is as follows:\n{job_description}\n"
    prompt += "\nReturn ONLY a numbered list of questions, no explanations."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800
            )
        )
        # Extract only the list of questions
        questions = []
        for line in response.text.splitlines():
            line = line.strip()
            if re.match(r'^[0-9]+[).]', line):
                q = re.sub(r'^[0-9]+[).]\s*', '', line)
                if q:
                    questions.append(q)
            elif line:
                questions.append(line)
        if not questions:
            questions = [response.text.strip()]
        # Save QuestionGeneration event
        user = get_current_user() if 'get_current_user' in globals() else None
        user_id = user.id if user else None
        qg = QuestionGeneration(
            user_id=user_id,
            technology=technology,
            job_description=job_description,
            num_questions=num_questions,
            level=level
        )
        db.session.add(qg)
        db.session.commit()
        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': f'Failed to generate questions: {e}'}), 500 