import streamlit as st
import ollama
import os
from datetime import datetime
import json
import tempfile

# Check for faster_whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    st.error("❌ **faster-whisper is not installed!**")
    st.code("pip install faster-whisper", language="bash")
    st.stop()

# --- App Configuration ---
st.set_page_config(
    page_title="Meeting Extractor (Local)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎤 Meeting Audio Extractor (Local)")
st.write("Upload meeting audio to get structured notes and task lists - **100% Local, No API Keys Needed!**")

# --- Session State Initialization ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "notes" not in st.session_state:
    st.session_state.notes = None
if "tasks" not in st.session_state:
    st.session_state.tasks = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("✅ **100% Local Processing**\n\nNo API keys required!\nUsing local Whisper + Ollama")
    
    st.divider()
    
    st.write("**Whisper Model Selection**")
    whisper_size = st.selectbox(
        "Whisper Model Size:",
        options=["base", "small", "medium", "large-v2", "large-v3"],
        index=1,
        help="Larger models = better accuracy but slower. 'base' is fastest, 'large-v3' is most accurate",
        key="whisper_size"
    )
    
    st.caption("💡 First run will download the model (~1-3GB)")
    
    st.divider()
    
    st.write("**Ollama Model Selection**")
    # Try to get available Ollama models
    try:
        available_models = ollama.list()
        model_names = [model['name'] for model in available_models.get('models', [])]
        
        # Filter for text models (not vision)
        text_models = [m for m in model_names if 'llava' not in m.lower()]
        
        if text_models:
            ollama_model_options = text_models
            # Prefer smaller/faster models
            preferred_order = ["mistral:latest", "llama3:latest", "gpt-oss:20b"]
            for pref in preferred_order:
                if pref in ollama_model_options:
                    default_ollama_index = ollama_model_options.index(pref)
                    break
            else:
                default_ollama_index = 0
        else:
            ollama_model_options = ["mistral:latest", "llama3:latest"]
            default_ollama_index = 0
            st.warning("No Ollama models found. Install one with: `ollama pull mistral:latest`")
    except Exception as e:
        ollama_model_options = ["mistral:latest", "llama3:latest"]
        default_ollama_index = 0
        st.warning("Could not connect to Ollama. Make sure it's running.")
    
    ollama_model = st.selectbox(
        "Ollama Model for Extraction:",
        options=ollama_model_options,
        index=default_ollama_index,
        help="Local Ollama model for structuring notes and extracting tasks",
        key="ollama_model"
    )
    
    st.divider()
    
    # Task extraction options
    st.write("**Extraction Options**")
    extract_tasks = st.checkbox("Extract Action Items", value=True, key="extract_tasks")
    extract_summary = st.checkbox("Generate Summary", value=True, key="extract_summary")
    extract_key_points = st.checkbox("Extract Key Points", value=True, key="extract_key_points")
    extract_decisions = st.checkbox("Extract Decisions", value=True, key="extract_decisions")
    
    st.divider()
    
    if st.button("🗑️ Clear All"):
        st.session_state.transcript = None
        st.session_state.notes = None
        st.session_state.tasks = None
        st.rerun()
    
    st.divider()
    st.write("**💡 Tips:**")
    st.caption("""
    - First transcription may take longer (model download)
    - Use 'base' or 'small' for faster processing
    - Use 'large-v3' for best accuracy
    - Make sure Ollama is running
    """)

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload Meeting Audio",
    type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac"],
    help="Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM, FLAC"
)

# --- Main Processing ---
if uploaded_file is not None:
    # Display file info
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"📁 File: {uploaded_file.name} | Size: {file_size_mb:.2f} MB")
    
    # Process button
    if st.button("🚀 Process Audio", type="primary", use_container_width=True):
        with st.spinner(f"🎙️ Transcribing audio with Whisper ({st.session_state.whisper_size})... This may take a moment..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_audio_path = tmp_file.name
                
                # Initialize Whisper model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading Whisper model...")
                progress_bar.progress(20)
                
                # Load Whisper model (will download on first use)
                try:
                    model = WhisperModel(
                        st.session_state.whisper_size,
                        device="cpu",  # Use "cuda" if you have GPU
                        compute_type="int8"  # Use "float16" for better quality if you have GPU
                    )
                except Exception as model_error:
                    st.error(f"❌ Failed to load Whisper model: {str(model_error)}")
                    st.info("💡 Try installing faster-whisper: `pip install faster-whisper`")
                    if 'tmp_audio_path' in locals():
                        os.remove(tmp_audio_path)
                    raise
                
                status_text.text("Transcribing audio...")
                progress_bar.progress(50)
                
                # Transcribe
                try:
                    segments, info = model.transcribe(tmp_audio_path, beam_size=5)
                except Exception as transcribe_error:
                    st.error(f"❌ Transcription failed: {str(transcribe_error)}")
                    st.info("💡 The audio file might be corrupted or in an unsupported format. Try converting to MP3 or WAV.")
                    if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
                    raise
                
                # Collect transcript
                transcript_parts = []
                for segment in segments:
                    transcript_parts.append(segment.text)
                
                full_transcript = " ".join(transcript_parts)
                
                progress_bar.progress(100)
                status_text.text("Transcription complete!")
                
                st.session_state.transcript = full_transcript
                
                # Clean up temp file
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Transcription complete!")
                
            except Exception as e:
                st.error(f"❌ Transcription error: {str(e)}")
                if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
    
    # Display transcript if available
    if st.session_state.transcript:
        with st.expander("📝 View Full Transcript", expanded=False):
            st.text_area("Transcript", st.session_state.transcript, height=200, disabled=True)
        
        # Extract structured notes and tasks
        if st.button("📋 Extract Notes & Tasks", type="primary", use_container_width=True):
            with st.spinner(f"🧠 Extracting structured information using {st.session_state.ollama_model}..."):
                try:
                    # Build extraction prompt
                    extraction_options = []
                    if st.session_state.extract_summary:
                        extraction_options.append("executive summary")
                    if st.session_state.extract_key_points:
                        extraction_options.append("key discussion points")
                    if st.session_state.extract_decisions:
                        extraction_options.append("decisions made")
                    if st.session_state.extract_tasks:
                        extraction_options.append("action items with assignees")
                    
                    extraction_request = ", ".join(extraction_options)
                    
                    prompt = f"""Analyze the following meeting transcript and extract structured information. 
Please provide a JSON response with the following structure:

{{
    "summary": "A brief executive summary of the meeting (2-3 sentences)",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "decisions": ["Decision 1", "Decision 2"],
    "action_items": [
        {{
            "task": "Task description",
            "assignee": "Person responsible (if mentioned, otherwise 'TBD')",
            "due_date": "Date mentioned (if any, otherwise 'TBD')",
            "priority": "High/Medium/Low (based on context)"
        }}
    ],
    "participants": ["Participant 1", "Participant 2"],
    "meeting_date": "Date mentioned in transcript (if any, otherwise 'Not specified')",
    "duration_estimate": "Estimated meeting duration based on transcript length"
}}

Focus on extracting: {extraction_request}

Meeting Transcript:
{st.session_state.transcript}

IMPORTANT: Respond with ONLY valid JSON. Do not include any markdown formatting, code blocks, or additional text. Just the raw JSON object."""

                    # Call Ollama
                    response = ollama.chat(
                        model=st.session_state.ollama_model,
                        messages=[
                            {"role": "system", "content": "You are a professional meeting analyst. Extract structured information from meeting transcripts and return ONLY valid JSON. Do not use markdown code blocks."},
                            {"role": "user", "content": prompt}
                        ],
                        options={
                            "temperature": 0.3
                        }
                    )
                    
                    # Parse response
                    response_text = response['message']['content'].strip()
                    
                    # Try to extract JSON if wrapped in markdown
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    # Remove any leading/trailing whitespace
                    response_text = response_text.strip()
                    
                    # Try to parse JSON
                    try:
                        extracted_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to find JSON object in the text
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            extracted_data = json.loads(json_match.group())
                        else:
                            raise ValueError("Could not parse JSON from response")
                    
                    st.session_state.notes = extracted_data
                    
                    st.success("✅ Extraction complete!")
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error parsing JSON response: {str(e)}")
                    st.code(response_text[:500], language="text")
                    st.info("💡 The model response couldn't be parsed as JSON. Try a different Ollama model or check the response above.")
                except Exception as e:
                    st.error(f"❌ Extraction error: {str(e)}")
                    st.info("💡 Make sure Ollama is running and the model is installed.")
    
    # Display structured results
    if st.session_state.notes:
        st.divider()
        st.header("📊 Structured Meeting Notes")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔑 Key Points", "✅ Action Items", "📄 Full Details"])
        
        with tab1:
            if st.session_state.notes.get("summary"):
                st.write("### Executive Summary")
                st.info(st.session_state.notes["summary"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.notes.get("meeting_date"):
                    st.metric("Meeting Date", st.session_state.notes["meeting_date"])
            with col2:
                if st.session_state.notes.get("duration_estimate"):
                    st.metric("Duration", st.session_state.notes["duration_estimate"])
            with col3:
                if st.session_state.notes.get("participants"):
                    st.metric("Participants", len(st.session_state.notes["participants"]))
        
        with tab2:
            if st.session_state.notes.get("key_points"):
                st.write("### Key Discussion Points")
                for i, point in enumerate(st.session_state.notes["key_points"], 1):
                    st.markdown(f"{i}. {point}")
            else:
                st.info("No key points extracted")
            
            if st.session_state.notes.get("decisions"):
                st.write("### Decisions Made")
                for i, decision in enumerate(st.session_state.notes["decisions"], 1):
                    st.success(f"✅ {decision}")
        
        with tab3:
            if st.session_state.notes.get("action_items"):
                st.write("### Action Items")
                
                # Create a table/cards for action items
                for i, item in enumerate(st.session_state.notes["action_items"], 1):
                    with st.container():
                        priority_color = {
                            "High": "🔴",
                            "Medium": "🟡",
                            "Low": "🟢"
                        }.get(item.get("priority", "Medium"), "⚪")
                        
                        st.markdown(f"""
                        **{priority_color} Task {i}:** {item.get('task', 'N/A')}
                        - **Assignee:** {item.get('assignee', 'TBD')}
                        - **Due Date:** {item.get('due_date', 'TBD')}
                        - **Priority:** {item.get('priority', 'Medium')}
                        """)
                        st.divider()
                
                # Download as markdown
                tasks_md = "## Action Items\n\n"
                for i, item in enumerate(st.session_state.notes["action_items"], 1):
                    tasks_md += f"{i}. **{item.get('task', 'N/A')}**\n"
                    tasks_md += f"   - Assignee: {item.get('assignee', 'TBD')}\n"
                    tasks_md += f"   - Due Date: {item.get('due_date', 'TBD')}\n"
                    tasks_md += f"   - Priority: {item.get('priority', 'Medium')}\n\n"
                
                st.download_button(
                    "📥 Download Action Items",
                    tasks_md,
                    file_name=f"action_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            else:
                st.info("No action items found in this meeting")
        
        with tab4:
            st.write("### Full Structured Data")
            st.json(st.session_state.notes)
            
            # Download full JSON
            json_str = json.dumps(st.session_state.notes, indent=2)
            st.download_button(
                "📥 Download Full JSON",
                json_str,
                file_name=f"meeting_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Sidebar participants list
        if st.session_state.notes.get("participants"):
            with st.sidebar:
                st.divider()
                st.write("**👥 Participants**")
                for participant in st.session_state.notes["participants"]:
                    st.write(f"• {participant}")

else:
    st.info("👆 Please upload an audio file to get started")
    
    # Instructions
    with st.expander("📖 How to use (100% Local)"):
        st.markdown("""
        ### ✅ No API Keys Required!
        
        This version uses:
        - **faster-whisper**: Local Whisper model for transcription
        - **Ollama**: Local LLM for extraction (already installed!)
        
        ### Steps:
        1. **Make sure Ollama is running** (you already have it installed)
        2. **Upload an audio file** (MP3, MP4, WAV, etc.)
        3. **Click "Process Audio"** to transcribe (first time will download the Whisper model)
        4. **Click "Extract Notes & Tasks"** to get structured information
        5. **Review and download** the structured notes
        
        ### Supported Audio Formats:
        - MP3, MP4, MPEG, MPGA
        - M4A, WAV, WEBM, FLAC
        
        ### Model Options:
        - **Whisper**: Choose size (base/small/medium/large-v3)
          - Smaller = faster, less accurate
          - Larger = slower, more accurate
        - **Ollama**: Use your installed models (mistral, llama3, etc.)
        
        ### First Run:
        - The first time you transcribe, faster-whisper will download the model (~1-3GB)
        - This only happens once per model size
        - Subsequent runs will be faster
        """)

