from flask import Blueprint, request, jsonify
import re
import google.generativeai as genai

questions_bp = Blueprint('questions', __name__)

@questions_bp.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    technology = data.get('technology', '').strip()
    job_description = data.get('job_description', '').strip()
    prompt = f"""
You are an expert interviewer. Generate a list of 5 relevant, challenging, and up-to-date interview questions for a {technology or 'General'} role.
"""
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
        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': f'Failed to generate questions: {e}'}), 500 