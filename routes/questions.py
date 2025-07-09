# from flask import Blueprint, request, jsonify, session
# import re
# import google.generativeai as genai
# from database import db, QuestionGeneration
# from utils import get_current_user
# from functools import wraps

# questions_bp = Blueprint('questions', __name__)

# # Authentication decorator for blueprint
# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'user' not in session:
#             # Clear any existing session data
#             session.clear()
#             return jsonify({'error': 'Unauthorized'}), 401
#         return f(*args, **kwargs)
#     return decorated_function

# @questions_bp.route('/generate_questions', methods=['POST'])
# @login_required
# def generate_questions():
#     data = request.get_json()
#     technology = data.get('technology', '').strip()
#     job_description = data.get('job_description', '').strip()
#     print(job_description)
#     num_questions = int(data.get('num_questions', 5))
#     level = data.get('level', '0-1').strip()
#     # Map experience level to descriptive text
#     experience_map = {
#         '0-1': 'entry-level (0-1 years of experience). Focus heavily on basics, core fundamentals, and simple problem-solving. Avoid advanced or scenario-based questions. Optionally, include very basic questions about simple optimizations or efficient solutions.',
#         '1-2': 'junior (1-2 years of experience). Include mostly basic and some intermediate questions, with a strong emphasis on practical scenarios and real-world applications. Slightly reduce pure fundamentals. Include simple questions about code optimization, performance improvements, and cost-effective solutions relevant to junior roles.',
#         '2-5': 'mid-level/team lead (2-5 years of experience). Cover a balanced mix of intermediate and some advanced topics, including practical application, troubleshooting, and real-world challenges. Also include questions about team leadership, mentoring, collaboration, handling team challenges, and practical approaches to optimization, performance tuning, and cost-saving techniques in projects.',
#         '5-10': 'senior (5-10 years of experience). Focus on advanced topics, architecture, design patterns, leadership, and complex problem-solving. Ask more project-oriented and solution architect-level questions, including system design, decision-making in large projects, and strategies for optimization, performance, and cost reduction at scale.',
#         '10+': 'expert (10+ years of experience). Emphasize deep expertise, strategic thinking, innovation, and high-level problem-solving. Ask project-oriented, solution architect, and expert-level questions, focusing on large-scale systems, technical vision, leadership in technology, and advanced strategies for cost optimization, performance, and resource management.'
#     }
#     experience_desc = experience_map.get(level, 'general')
#     prompt = f"""
# You are an expert interviewer. Generate a list of {num_questions} relevant, challenging, and up-to-date interview questions for a {technology or 'General'} role.
# The candidate is {experience_desc}
# """
#     if job_description:
#         prompt += f"\nThe job description is as follows:\n{job_description}\n"
#     prompt += ("\nMake each question short, direct, and use simple language. Avoid multi-part or overly detailed questions."
#                 "\nReturn ONLY a numbered list of questions, no explanations.")
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 max_output_tokens=800
#             )
#         )
#         # Extract only the list of questions
#         questions = []
#         for line in response.text.splitlines():
#             line = line.strip()
#             if re.match(r'^[0-9]+[).]', line):
#                 q = re.sub(r'^[0-9]+[).]\s*', '', line)
#                 if q:
#                     questions.append(q)
#             elif line:
#                 questions.append(line)
#         if not questions:
#             questions = [response.text.strip()]
#         # Save QuestionGeneration event
#         user = get_current_user() if 'get_current_user' in globals() else None
#         user_id = user.id if user else None
#         qg = QuestionGeneration(
#             user_id=user_id,
#             technology=technology,
#             job_description=job_description,
#             num_questions=num_questions,
#             level=level
#         )
#         db.session.add(qg)
#         db.session.commit()
#         return jsonify({'questions': questions})
#     except Exception as e:
#         return jsonify({'error': f'Failed to generate questions: {e}'}), 500 

from flask import Blueprint, request, jsonify, session
import re
import google.generativeai as genai
from database import db, QuestionGeneration, IQGFileUpload
from utils import get_current_user
from functools import wraps

questions_bp = Blueprint('questions', __name__)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            session.clear()
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@questions_bp.route('/generate_questions', methods=['POST'])
@login_required
def generate_questions():
    data = request.get_json()
    job_description = data.get('job_description', '').strip()
    num_questions = int(data.get('num_questions', 5))
    level = data.get('level', '0-1').strip()

    # Updated experience map descriptions
    experience_map = {
    '0-1': (
        "Entry-level (0-1 years). Focus on core programming fundamentals, basic syntax, simple logic, "
        "and straightforward tasks. Include easy problem-solving and introductory concepts. "
        "Avoid advanced topics, multi-step troubleshooting, or architectural discussions."
    ),
    '1-2': (
        "Junior (1-2 years). Emphasize foundational knowledge with some light intermediate concepts. "
        "Include questions on writing clean code, debugging small issues, and minor optimizations.The questions should NOT be scenario based but to the point"
    ),
    '2-5': (
        "Mid-level (2-5 years). Cover a balanced mix of intermediate and advanced topics, including "
        "practical applications, performance tuning, and collaborating with teams. "
        "Include questions on complex systems, architectural design fundamentals, and coordinating small multi-team efforts. The question should test the technical acumen of the candidate"
    ),
    '5-10': (
        "Senior (5-10 years). Focus on advanced problem-solving, full-scale architectural design, leadership, "
        "and decision-making in large, complex projects. Include system design, scaling strategies, and techniques "
        "for cost and performance optimization, along with managing cross-team collaborations."
    ),
    '10+': (
        "Expert (10+ years). Emphasize strategic thinking, innovation, and deep technical expertise. "
        "Include questions on large-scale systems, enterprise architecture, technical vision, advanced resource management, "
        "and leading organization-wide technology initiatives that involve multiple teams and business units."
    )
}



    experience_desc = experience_map.get(level, 
        "general. Include a balanced mix of basic, intermediate, and advanced questions depending on context."
    )

    # Build prompt purely on level + job description
    prompt = (
        f"You are an expert interviewer. Generate a list of {num_questions} relevant, "
        f"challenging, and up-to-date interview questions for this position. "
        f"The candidate is {experience_desc}"
    )
    if job_description:
        prompt += f"\nThe job description is as follows:\n{job_description}\n"

    prompt += (
        "\nMake each question short, direct, and use simple language. "
        "Avoid multi-part or overly detailed questions."
        "\nReturn ONLY a numbered list of questions, no explanations."
    )

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800
            )
        )

        # Extract clean list of questions
        questions = []
        for line in response.text.splitlines():
            line = line.strip()
            if re.match(r'^\d+[).]', line):
                question = re.sub(r'^\d+[).]\s*', '', line)
                if question:
                    questions.append(question)
            elif line:
                questions.append(line)

        if not questions:
            questions = [response.text.strip()]

        # Log in database
        user = get_current_user() if 'get_current_user' in globals() else None
        user_id = user.id if user else None
        qg = QuestionGeneration(
            user_id=user_id,
            technology=None,
            job_description=job_description,
            num_questions=num_questions,
            level=level
        )
        db.session.add(qg)
        db.session.commit()

        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': f'Failed to generate questions: {e}'}), 500

@questions_bp.route('/get_iqg_file/<int:file_id>', methods=['GET'])
def get_iqg_file(file_id):
    file = IQGFileUpload.query.get(file_id)
    if not file:
        return jsonify({'content': ''}), 404
    # Try to decode as UTF-8 text (for .txt), or extract text from .docx if needed
    import os
    import tempfile
    content = ''
    filename = file.original_filename.lower()
    if filename.endswith('.txt'):
        try:
            content = file.file_data.decode('utf-8', errors='replace')
        except Exception:
            content = ''
    elif filename.endswith('.docx'):
        try:
            import mammoth
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(file.file_data)
                tmp.flush()
                with open(tmp.name, 'rb') as docx_file:
                    result = mammoth.extract_raw_text(docx_file)
                    content = result.value
            os.unlink(tmp.name)
        except Exception:
            content = ''
    else:
        # Unknown file type
        content = ''
    return jsonify({'content': content})
