# Speech-to-Text Transcription App

A Flask-based web application that can transcribe speech from audio files using Facebook's Wav2Vec2 pre-trained model. The application features a soothing light green theme and attractive animations.

## Features

- Upload audio files for transcription
- Try demo audio samples
- Beautiful UI with soothing green theme
- Animated sound wave effect
- Responsive design for mobile and desktop
- Support for WAV, MP3, OGG, and FLAC audio formats

## Tech Stack

- **Backend**: Flask (Python)
- **Speech Recognition**: Facebook's Wav2Vec2 model via Hugging Face Transformers
- **Frontend**: HTML, CSS, JavaScript
- **Audio Processing**: PyTorch and Torchaudio

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Add demo audio files:
   - Create `static/uploads/demos/` directory
   - Add sample audio files like `welcome_message.wav`, `short_speech.wav`, and `counting.wav`

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`
3. Upload an audio file or try one of the demo samples
4. View the transcription result

## Project Structure

```
speech-to-text-app/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── static/                 # Static files
│   └── uploads/            # Uploaded audio files
│       └── demos/          # Demo audio files
└── templates/
    └── index.html          # Main application template
```

## Adding Demo Audio Files

The application comes with placeholders for three demo audio files. You'll need to add these files yourself:

1. Create the directory: `static/uploads/demos/`
2. Add the following files:
   - `welcome_message.wav`: A short welcome message
   - `short_speech.wav`: A sample speech (30 seconds or less)
   - `counting.wav`: Someone counting from 1 to 10

You can record these yourself or download samples from free audio libraries. If you want to use different files, update the `DEMO_AUDIO_FILES` list in `app.py`.

## Model Information

This application uses Facebook's Wav2Vec2 model, which is a powerful speech recognition model trained on 960 hours of unlabeled speech from LibriSpeech. The model converts audio waveforms into contextual representations that can be translated into text.

## Limitations

- Maximum file size is limited to 16MB
- Best results are achieved with clear audio and minimal background noise
- Processing time may vary depending on your hardware
- Currently only supports English language audio

## License

This project is released under the MIT License.