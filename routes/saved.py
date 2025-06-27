from flask import Blueprint, request, jsonify
import os, glob, json, datetime
from utils import SAVED_FILES_FOLDER

saved_bp = Blueprint('saved', __name__)

@saved_bp.route('/save_analysis', methods=['POST'])
def save_analysis():
    data = request.get_json()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_{timestamp}.json"
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'success', 'filename': filename})

@saved_bp.route('/list_saved_analyses', methods=['GET'])
def list_saved_analyses():
    files = glob.glob(os.path.join(SAVED_FILES_FOLDER, 'analysis_*.json'))
    files.sort(reverse=True)
    result = []
    for f in files:
        fname = os.path.basename(f)
        ts = fname.replace('analysis_', '').replace('.json', '')
        result.append({'filename': fname, 'timestamp': ts})
    return jsonify(result)

@saved_bp.route('/get_saved_analysis/<filename>', methods=['GET'])
def get_saved_analysis(filename):
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

@saved_bp.route('/delete_saved_analysis/<filename>', methods=['DELETE'])
def delete_saved_analysis(filename):
    filepath = os.path.join(SAVED_FILES_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    os.remove(filepath)
    return jsonify({'status': 'deleted'}) 