from .auth import auth_bp
from .resume import resume_bp
from .audio import audio_bp
from .questions import questions_bp
from .saved import saved_bp

def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(resume_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(questions_bp)
    app.register_blueprint(saved_bp) 