from flask import Blueprint, request, jsonify
import os, uuid
from utils import transcribe_audio_with_sarvam, separate_hr_candidate_with_openai, evaluate_technical_proficiency_with_openai

audio_bp = Blueprint('audio', __name__)

@audio_bp.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files.get('audio')
    technology = request.form.get('technology', 'General')
    tech_questions = request.form.get('tech_questions', '').strip()
    response = {}
    if audio_file and audio_file.filename:
        audio_path = os.path.join('uploads', str(uuid.uuid4()) + '_' + audio_file.filename)
        audio_file.save(audio_path)
        transcript = transcribe_audio_with_sarvam(audio_path)
        if isinstance(transcript, str) and transcript.startswith('Error'):
            response['audio_error'] = transcript
        elif isinstance(transcript, str) and transcript.strip():
            response['transcription'] = transcript
            separated_dialog = separate_hr_candidate_with_openai(transcript)
            response['separated_dialog'] = separated_dialog
            candidate_lines = []
            for line in separated_dialog.splitlines():
                if line.strip().startswith('Candidate:'):
                    candidate_lines.append(line.replace('Candidate:', '').strip())
            candidate_text = '\n'.join(candidate_lines)
            tech_eval = evaluate_technical_proficiency_with_openai(
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