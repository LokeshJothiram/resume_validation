# AI Talent Acquisition Assistant

Effortlessly evaluate candidates and save your analyses with AI.

## Overview
This web application streamlines the technical hiring process by allowing you to:
- Upload and analyze candidate resumes (PDF/DOCX)
- Upload and analyze technical interview audio
- Get AI-powered scoring and explanations for both resumes and interviews
- Save and manage analysis results for future reference
- Secure access with a login system

## Features
- **Resume Match:** Upload one or more resumes and a job description. The app uses AI to score and explain how well each resume matches the job description.
- **Technical Proficiency:** Upload an interview audio file and (optionally) technology questions. The app transcribes, analyzes, and scores the candidate's technical proficiency.
- **Saved Analyses:** Save any analysis result to your local server. View or delete saved analyses from the "Saved Analyses" tab.
- **Modern UI:** Responsive, user-friendly interface with tabbed navigation, modals, and toast notifications.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd resume_validation
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

### 5. Run the application
```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
1. **Login** with one of the hardcoded users (see `users.py`).
2. Use the **Resume Match** or **Technical Proficiency** tabs to upload files and analyze candidates.
3. Click **Save** to store analysis results. View or delete them in the **Saved Analyses** tab.

## File Structure
- `app.py` - Main Flask backend
- `users.py` - User credentials
- `templates/` - HTML templates
- `static/` - CSS, images, and JS
- `uploads/` - Temporary file uploads
- `saved_files/` - Saved analysis results

## Notes
- This app uses Google Gemini and SarvamAI APIs for AI analysis. You must provide valid API keys.
- For demo purposes, user authentication is hardcoded. For production, use a secure authentication system.

## License
MIT License 