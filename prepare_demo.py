import os
import torchaudio
import torch
import numpy as np
from scipy.io import wavfile
from gtts import gTTS

"""
This script creates demo audio files for the Speech-to-Text application.
It generates three sample audio files using gTTS (Google Text-to-Speech).
"""

# Create demo directory if it doesn't exist
demo_dir = os.path.join('static', 'uploads', 'demos')
os.makedirs(demo_dir, exist_ok=True)

# Sample texts for demo audio files
demo_texts = {
    'welcome_message.wav': 'Welcome to the Speech to Text transcription app. Upload your audio file to get started.',
    'short_speech.wav': 'Artificial intelligence is transforming how we interact with technology. Speech recognition is one example of AI that helps computers understand human language.',
    'counting.wav': 'One, two, three, four, five, six, seven, eight, nine, ten.'
}

# Generate audio files using gTTS
for filename, text in demo_texts.items():
    output_path = os.path.join(demo_dir, filename)
    
    # Generate audio using Google Text-to-Speech
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    
    print(f"Created demo file: {filename}")

print("\nDemo files have been created successfully!")
print("You can now run the main application: python app.py")