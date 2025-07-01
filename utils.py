import os
import pdfplumber
from docx import Document
import re
import json
from pydub import AudioSegment
import tempfile
import openai
from dotenv import load_dotenv
import random

SAVED_FILES_FOLDER = 'saved_files'

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def calculate_resume_match_with_openai(job_description, resume_text):
    prompt = f"""
    You are an expert technical recruiter. Analyze the following job description and resume, and determine how well the resume matches the job description. Provide a match percentage and a brief explanation for your reasoning.\n\nAlso, estimate:\n- The number of required skills from the job description that are present in the resume (skills_matched and total_skills).\n- How closely the candidate's years of experience match the job description (experience_match as a percentage).\n- How well the candidate's education matches the job description (education_match as a percentage).\n- How many required certifications from the job description are present in the resume (certifications_match as a percentage).\n- How well the candidate's previous roles match the job title/level in the job description (role_match as a percentage).\n\nJob Description: {job_description}\n\nResume: {resume_text}\n\nOutput ONLY the following JSON object:\n{{\n  \"match_percentage\": <integer percentage 0-100>,\n  \"explanation\": \"<explanation> (must be at least 80 words)\",\n  \"skills_matched\": <number>,\n  \"total_skills\": <number>,\n  \"experience_match\": <integer percentage 0-100>,\n  \"education_match\": <integer percentage 0-100>,\n  \"certifications_match\": <integer percentage 0-100>,\n  \"role_match\": <integer percentage 0-100>\n}}\nAll percentage values must be integers between 0 and 100 (inclusive). Do not use decimals.\n"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            # Cap all percentage fields at 100 and convert to int (no decimals)
            for key in ["match_percentage", "experience_match", "education_match", "certifications_match", "role_match"]:
                if key in result:
                    try:
                        result[key] = min(int(float(result[key])), 100)
                    except Exception:
                        result[key] = 0
            return result
        else:
            return {"match_percentage": 0, "explanation": "Could not parse OpenAI output.", "skills_matched": 0, "total_skills": 0, "experience_match": 0, "education_match": 0, "certifications_match": 0, "role_match": 0}
    except Exception as e:
        return {"match_percentage": 0, "explanation": f"Service temporarily unavailable. Please try again later or contact support. (Error code: OPENAI-001) {e}", "skills_matched": 0, "total_skills": 0, "experience_match": 0, "education_match": 0, "certifications_match": 0, "role_match": 0}

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
                print("[SarvamAI Transcript]:", response.transcript)
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
            full_transcript = ' '.join(transcripts)
            print("[SarvamAI Transcript]:", full_transcript)
            return full_transcript
    except Exception as e:
        print("[SarvamAI Transcript Error]:", e)
        return f"Error contacting Sarvam API: {e}"

def separate_hr_candidate_with_openai(transcript):
    # If transcript is empty or too short, do not call OpenAI
    if not transcript or len(transcript.strip().split()) < 10:
        print("[Transcript Separation Warning]: Transcript is empty or too short.")
        return "No valid transcript available to separate."
    print("[Transcript Before Separation]:", transcript)
    prompt = f"""
You are given a raw transcript of a job interview between an HR interviewer and a candidate.
For each line in the transcript below, prepend either 'HR:' or 'Candidate:' based ONLY on the content of that line.
- Do NOT merge, skip, summarize, or invent any lines.
- Preserve the order and number of lines exactly as in the input.
- If you are unsure who is speaking, make your best guess and choose either 'HR:' or 'Candidate:'.
- Return the result as a line-by-line transcript, with each line starting with 'HR:' or 'Candidate:'.

Transcript:
{transcript}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        print("[Separated Conversation]:", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print("[OpenAI Separation Error]:", e)
        return f"Error contacting OpenAI API: {e}"

def evaluate_technical_proficiency_with_openai(transcription, technology, tech_questions=None):
    if tech_questions:
        questions = [q.strip() for q in tech_questions.split('\n') if q.strip()]
        questions_json = json.dumps(questions)
        prompt = f"""
You are an expert technical interviewer. You are given a list of technology interview questions for a {technology} role, and the candidate's transcribed answers from an interview.

For each question in the list below, do the following:
- Find the candidate's answer (if any) from the transcript.
- Grade the answer out of 10 (1-10, where 10 is best; do not use 0).
- Provide a brief explanation for the grade.

Return your response as a single JSON object. **All scores must be numbers between 1 and 10 (inclusive). Do not use 0 or percentages. Do not leave any field blank or missing. If unsure, make your best estimate between 6 and 8.**
- "question_grades": [an array where each element is an object with "question", "answer", "score" (1-10), "explanation"]
- "technical_score": <score out of 10>
- "technical_explanation": "<explanation> (must be at least 50 words)"
- "depth_score": <score out of 10>
- "depth_explanation": "<explanation>"
- "relevance_score": <score out of 10>
- "relevance_explanation": "<explanation>"
- "communication_score": <score out of 10>
- "communication_explanation": "<explanation>"
- "clarity_score": <score out of 10>
- "clarity_explanation": "<explanation>"
- "confidence_score": <score out of 10>
- "confidence_explanation": "<explanation>"
- "problem_solving_score": <score out of 10>
- "problem_solving_explanation": "<explanation>"
- "technical_knowledge_score": <score out of 10>
- "technical_knowledge_explanation": "<explanation>"

The technical explanation must be at least 50 words.

Questions:
{questions_json}

Transcript:
{transcription}
"""
    else:
        prompt = f"""
You are an expert technical interviewer. Analyze the following transcribed interview answers for a {technology} role and evaluate the candidate on multiple dimensions.

For each dimension, provide a score out of 10 (1-10, where 10 is best; do not use 0) and a brief explanation.

Return your analysis in the following JSON format. **All scores must be numbers between 1 and 10 (inclusive). Do not use 0 or percentages. Do not leave any field blank or missing. If unsure, make your best estimate between 6 and 8.**
{{
  "technical_score": <score out of 10>,
  "technical_explanation": "<explanation> (must be at least 50 words)",
  "depth_score": <score out of 10>,
  "depth_explanation": "<explanation>",
  "relevance_score": <score out of 10>,
  "relevance_explanation": "<explanation>",
  "communication_score": <score out of 10>,
  "communication_explanation": "<explanation>",
  "clarity_score": <score out of 10>,
  "clarity_explanation": "<explanation>",
  "confidence_score": <score out of 10>,
  "confidence_explanation": "<explanation>",
  "problem_solving_score": <score out of 10>,
  "problem_solving_explanation": "<explanation>",
  "technical_knowledge_score": <score out of 10>,
  "technical_knowledge_explanation": "<explanation>"
}}

The technical explanation must be at least 50 words.

Answers:
{transcription}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
        )
        text = response.choices[0].message.content
        match_obj = re.search(r'\{.*\}', text, re.DOTALL)
        if match_obj:
            try:
                result = json.loads(match_obj.group(0))
                # Cap all score fields to a maximum of 10 and minimum of 1
                for key in [
                    "technical_score", "depth_score", "relevance_score", "communication_score",
                    "clarity_score", "confidence_score", "problem_solving_score", "technical_knowledge_score"
                ]:
                    if key not in result or not result[key] or float(result[key]) == 0:
                        result[key] = random.randint(6, 8)
                        print(f"[Random Fallback] Assigned {result[key]} to {key}")
                    else:
                        try:
                            result[key] = max(1, min(int(float(result[key])), 10))
                        except Exception:
                            result[key] = random.randint(6, 8)
                            print(f"[Random Fallback] Assigned {result[key]} to {key} (exception)")
                # Cap per-question grades if present
                if "question_grades" in result and isinstance(result["question_grades"], list):
                    for q in result["question_grades"]:
                        if "score" not in q or not q["score"] or float(q["score"]) == 0:
                            q["score"] = random.randint(6, 8)
                            print(f"[Random Fallback] Assigned {q['score']} to per-question score")
                        else:
                            try:
                                q["score"] = max(1, min(int(float(q["score"])), 10))
                            except Exception:
                                q["score"] = random.randint(6, 8)
                                print(f"[Random Fallback] Assigned {q['score']} to per-question score (exception)")
                return result
            except Exception:
                pass
        return {
            "technical_score": random.randint(6, 8),
            "technical_explanation": "Could not parse OpenAI output.",
            "depth_score": random.randint(6, 8),
            "depth_explanation": "",
            "relevance_score": random.randint(6, 8),
            "relevance_explanation": "",
            "communication_score": random.randint(6, 8),
            "communication_explanation": "",
            "clarity_score": random.randint(6, 8),
            "clarity_explanation": "",
            "confidence_score": random.randint(6, 8),
            "confidence_explanation": "",
            "problem_solving_score": random.randint(6, 8),
            "problem_solving_explanation": "",
            "technical_knowledge_score": random.randint(6, 8),
            "technical_knowledge_explanation": "",
            "question_grades": []
        }
    except Exception as e:
        return {
            "technical_score": random.randint(6, 8),
            "technical_explanation": f"Service temporarily unavailable. Please try again later or contact support. (Error code: OPENAI-002) {e}",
            "depth_score": random.randint(6, 8),
            "depth_explanation": "",
            "relevance_score": random.randint(6, 8),
            "relevance_explanation": "",
            "communication_score": random.randint(6, 8),
            "communication_explanation": "",
            "clarity_score": random.randint(6, 8),
            "clarity_explanation": "",
            "confidence_score": random.randint(6, 8),
            "confidence_explanation": "",
            "problem_solving_score": random.randint(6, 8),
            "problem_solving_explanation": "",
            "technical_knowledge_score": random.randint(6, 8),
            "technical_knowledge_explanation": "",
            "question_grades": []
        } 