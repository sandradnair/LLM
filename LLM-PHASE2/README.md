# 🎙️ Meeting Notes & Action Items Extractor

A Streamlit application that transcribes meeting audio files and generates structured summaries with action items using local AI models (Whisper and Ollama).

## 🌟 Features

- Audio file transcription using OpenAI's Whisper model
- Automated generation of meeting summaries and action items
- Local execution using Ollama for AI processing
- Support for multiple audio formats (MP3, WAV, M4A)
- Real-time streaming of AI-generated notes
- User-friendly web interface

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.8+**
2. **ffmpeg** (required for audio processing)
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
3. **Ollama**: Download and install from [ollama.com](https://ollama.com)

## 🚀 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd meeting-notes-app
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Start the Ollama server:
```bash
ollama serve
```

4. Pull the required language model (in a new terminal):
```bash
ollama pull llama3
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (typically `http://localhost:8501`)

3. Upload an audio file (supported formats: .mp3, .wav, .m4a)

4. Wait for the transcription to complete

5. Click "Generate Notes" to get the AI-generated summary and action items
