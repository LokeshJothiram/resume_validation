from flask import Blueprint, request, jsonify
import re
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

questions_bp = Blueprint('questions', __name__)

@questions_bp.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    technology = data.get('technology', '').strip()
    job_description = data.get('job_description', '').strip()
    num_questions = int(data.get('num_questions', 5))
    if num_questions > 20:
        num_questions = 20
    if num_questions < 1:
        num_questions = 1
    prompt = f"""
You are an expert interviewer. Generate a list of {num_questions} relevant, challenging, and up-to-date interview questions for a {technology or 'General'} role.
"""
    if job_description:
        prompt += f"\nThe job description is as follows:\n{job_description}\n"
    prompt += "\nReturn ONLY a numbered list of questions, no explanations."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        questions = []
        for line in response.choices[0].message.content.splitlines():
            line = line.strip()
            if re.match(r'^[0-9]+[).]', line):
                q = re.sub(r'^[0-9]+[).]\s*', '', line)
                if q:
                    questions.append(q)
            elif line:
                questions.append(line)
        if not questions:
            questions = [response.choices[0].message.content.strip()]
        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': f'Failed to generate questions: {e}'}), 500 