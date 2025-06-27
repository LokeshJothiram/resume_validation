import os
import pdfplumber
from docx import Document
import google.generativeai as genai
import re
import json
from pydub import AudioSegment
import tempfile

SAVED_FILES_FOLDER = 'saved_files'

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

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

def split_audio(audio_path, chunk_length_ms=30000):
    """
    Splits an audio file into chunks of chunk_length_ms milliseconds (default 30 seconds).
    Returns a list of file paths to the temporary chunk files.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_file.name, format="wav")
        chunks.append(temp_file.name)
    return chunks

def transcribe_audio_with_sarvam(audio_path, model="saarika:v2.5", language_code="en-IN"):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        from app import client
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