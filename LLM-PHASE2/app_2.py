import streamlit as st
import whisper
import requests
import json
import os
import tempfile

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# You can change this to "mistral", "phi3", etc. if you have them pulled
OLLAMA_MODEL = "llama3" 
PROMPT_TEMPLATE = """
You are an expert meeting assistant. Your task is to analyze the following meeting transcript and provide a structured summary in Markdown format.

The output must include two distinct sections:
1.  **Meeting Summary:** A concise overview of the key topics discussed, decisions made, and main outcomes.
2.  **Action Items:** A checklist of all tasks assigned during the meeting. For each task, identify the person responsible if mentioned.

Here is the transcript:
---
{transcription}
---
"""

# --- Model Loading ---
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model and caches it."""
    try:
        model = whisper.load_model("base")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.stop()

# --- API Communication ---
def generate_notes_with_ollama(transcription, placeholder):
    """Generates notes and action items using the Ollama API, streaming the response."""
    prompt = PROMPT_TEMPLATE.format(transcription=transcription)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }
    full_response = ""
    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = json.loads(chunk)
                    if not decoded_chunk.get("done"):
                        token = decoded_chunk.get("response", "")
                        full_response += token
                        placeholder.markdown(full_response + " ▌")
            placeholder.markdown(full_response) # Final update without cursor
        return full_response
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to Ollama API: {e}\\n\\nPlease ensure the Ollama server is running and accessible at `{OLLAMA_API_URL}`."
        placeholder.error(error_message)
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON from Ollama: {e}"
        placeholder.error(error_message)
    return None


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Meeting Notes Extractor")
st.title("🎙️ Meeting Notes & Action Item Extractor")
st.markdown("Upload a meeting audio file to transcribe it and generate a structured summary with action items using local AI models.")

st.sidebar.header("Setup Instructions")
st.sidebar.info(
    "**For this app to work, you need to:**\n"
    "1. **Install ffmpeg:** `sudo apt update && sudo apt install ffmpeg` (on Debian/Ubuntu) or `brew install ffmpeg` (on macOS).\n"
    "2. **Install Ollama:** Follow the instructions on [ollama.com](https://ollama.com).\n"
    "3. **Run Ollama Server:** Open a terminal and run `ollama serve`.\n"
    "4. **Pull a model:** In another terminal, run `ollama pull llama3` (or another model like `mistral`).\n"
    "5. **Install Python packages:** `pip install -r requirements.txt`."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. Make sure your Ollama server is running.\n"
    "2. Upload an audio file (`.mp3`, `.wav`, `.m4a`).\n"
    "3. The app will automatically transcribe the audio.\n"
    "4. Click **Generate Notes** to get the summary and action items."
)

# Load Whisper model
with st.spinner("Loading speech-to-text model... This might take a moment on the first run."):
    whisper_model = load_whisper_model()

# File Uploader
uploaded_file = st.file_uploader(
    "**Upload your audio file**",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio_file:
        tmp_audio_file.write(uploaded_file.getvalue())
        tmp_audio_path = tmp_audio_file.name

    col1, col2 = st.columns(2)

    try:
        with col1:
            st.subheader("Transcription")
            with st.spinner("Transcribing audio... This may take a while for long files."):
                transcription_result = whisper_model.transcribe(tmp_audio_path, fp16=False)
                transcription_text = transcription_result["text"]
                st.text_area("Full Transcript", transcription_text, height=400, key="transcript")

        with col2:
            st.subheader("Meeting Notes & Action Items")
            if st.button("Generate Notes", key="generate", type="primary"):
                with st.spinner(f"Generating notes with Ollama model: `{OLLAMA_MODEL}`..."):
                    notes_placeholder = st.empty()
                    generate_notes_with_ollama(transcription_text, notes_placeholder)

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

else:
    st.warning("Please upload an audio file to begin processing.")
