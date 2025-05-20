from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model and processor
print("Loading Wav2Vec2 model... (this may take a moment)")
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def transcribe_audio(audio_path):
    """Transcribe audio file using Wav2Vec2 model"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process the audio
        input_values = processor(waveform.squeeze().numpy(), 
                                sampling_rate=sample_rate, 
                                return_tensors="pt").input_values
        
        # Get the transcription
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode the prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription

    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise

# Sample audio files for demo
DEMO_AUDIO_FILES = [
    {"name": "welcome_message.wav", "title": "Welcome Message"},
    {"name": "short_speech.wav", "title": "Short Speech Sample"},
    {"name": "counting.wav", "title": "Counting 1-10"}
]

@app.route('/')
def index():
    return render_template('index.html', demo_files=DEMO_AUDIO_FILES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Add a small delay to simulate processing for demo purposes
            time.sleep(1)
            
            transcription = transcribe_audio(file_path)
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': file_path.replace('static/', ''),
                'transcription': transcription
            })
        except Exception as e:
            return jsonify({'error': f'Error during transcription: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/transcribe-demo/<filename>')
def transcribe_demo(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demos', filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Demo file not found'}), 404
    
    try:
        # Add a small delay to simulate processing for demo purposes
        time.sleep(1)
        
        transcription = transcribe_audio(file_path)
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'uploads/demos/{filename}',
            'transcription': transcription
        })
    except Exception as e:
        return jsonify({'error': f'Error during transcription: {str(e)}'}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create demo files directory
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'demos'), exist_ok=True)
    
    # Check if demo files exist
    demo_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'demos')
    demo_files_exist = all(os.path.exists(os.path.join(demo_dir, demo['name'])) for demo in DEMO_AUDIO_FILES)
    
    if not demo_files_exist:
        print("\nDemo audio files not found. Please run 'python prepare_demo.py' to generate them.")
        print("Continuing without demo files...\n")
    
    print("Starting Speech-to-Text Web Application...")
    print("Open your browser and navigate to: http://127.0.0.1:5001/")
    app.run(host='0.0.0.0',port=5001,debug=True)