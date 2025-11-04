import streamlit as st
import ollama
from PIL import Image
import io

# --- App Configuration ---
st.set_page_config(
    page_title="Multimodal Assistant",
    layout="wide"
)

st.title("🤖 Your Local Multi-Modal Assistant")
st.write("Chat with images or text using your local Ollama models!")

# --- Session State Initialization ---
# This is to keep track of the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "mistral:latest"  # Default to text model (lower memory)
if "mode" not in st.session_state:
    st.session_state.mode = "text"  # Default to text mode (lower memory)

# Memory-friendly notice
if st.session_state.mode == "text":
    st.success("✅ **Text Mode Active** - Using memory-efficient models (mistral/llama3)")
else:
    st.info("💡 **Tip:** If you encounter memory errors, switch to **Text mode** in the sidebar for lower memory usage")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selection
    mode = st.radio(
        "**Mode:**",
        options=["vision", "text"],
        index=0 if st.session_state.mode == "vision" else 1,
        format_func=lambda x: "🖼️ Vision (with images)" if x == "vision" else "💬 Text Only",
        key="mode_selector"
    )
    st.session_state.mode = mode
    
    st.divider()
    st.write("**Model Selection**")
    
    # Try to get available models dynamically
    try:
        available_models = ollama.list()
        model_names = [model['name'] for model in available_models.get('models', [])]
        
        if mode == "vision":
            # Filter for vision models (llava)
            vision_models = [m for m in model_names if 'llava' in m.lower()]
            
            if vision_models:
                model_options = vision_models
                # Prefer smaller models first
                preferred_order = ["llava:7b", "llava"]
                for pref in preferred_order:
                    if pref in model_options:
                        default_index = model_options.index(pref)
                        break
                else:
                    # Set default to first available
                    default_index = 0
            else:
                model_options = ["llava:7b", "llava:latest"]
                default_index = 0
                st.warning("⚠️ No Llava models found. Vision mode requires a Llava model.")
                st.info("💡 Switch to **Text mode** to use your existing models (mistral, llama3, etc.)")
        else:
            # Text-only models
            text_models = [m for m in model_names if 'llava' not in m.lower()]
            
            if text_models:
                model_options = text_models
                # Prefer smaller models first
                preferred_order = ["mistral:latest", "llama3:latest", "gpt-oss:20b"]
                for pref in preferred_order:
                    if pref in model_options:
                        default_index = model_options.index(pref)
                        break
                else:
                    default_index = 0
            else:
                model_options = ["mistral:latest", "llama3:latest"]
                default_index = 0
                st.warning("No text models found.")
    except Exception as e:
        # Fallback
        if mode == "vision":
            model_options = ["llava:latest"]
        else:
            model_options = ["mistral:latest", "llama3:latest", "gpt-oss:20b"]
        default_index = 0
        st.warning("Could not load models. Make sure Ollama is running.")
    
    if mode == "vision":
        st.warning("⚠️ **Vision models require more memory**")
        st.write("💡 **Tips if you get memory errors:**")
        st.markdown("""
        1. **Switch to Text mode** (recommended) - uses less memory
        2. **Try smaller Llava model:** `ollama pull llava:7b`
        3. **Close other applications** to free up RAM
        4. **Use smaller images** (resize to 512x512 or smaller)
        """)
    
    selected_model = st.selectbox(
        f"Choose a {'vision' if mode == 'vision' else 'text'} model:",
        options=model_options,
        index=default_index,
        key="model_selector"
    )
    st.session_state.model = selected_model
    
    st.info(f"**Current model:** {selected_model}")
    
    # Show model info if available
    try:
        available_models = ollama.list()
        for model in available_models.get('models', []):
            if model['name'] == selected_model:
                size_gb = model.get('size', 0) / (1024**3)  # Convert to GB
                st.caption(f"Size: ~{size_gb:.1f} GB")
                break
    except:
        pass
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Models"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# --- UI Components ---
if st.session_state.mode == "vision":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    prompt_text = st.chat_input("Ask a question about the image...")
    
    # Display the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)
else:
    uploaded_file = None
    prompt_text = st.chat_input("Ask a question or start a conversation...")

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---
if prompt_text:
    # Vision mode: requires image
    if st.session_state.mode == "vision" and uploaded_file is None:
        st.warning("⚠️ Please upload an image before asking a question in vision mode.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Prepare for assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare messages with chat history
            messages_for_ollama = []
            # Add recent chat history (excluding the current message we just added)
            for msg in st.session_state.messages[:-1][-10:]:
                msg_dict = {
                    'role': msg['role'],
                    'content': msg['content']
                }
                messages_for_ollama.append(msg_dict)

            # Call the Ollama model
            try:
                with st.spinner("Thinking..."):
                    if st.session_state.mode == "vision" and uploaded_file is not None:
                        # Vision mode: include image
                        image_bytes = uploaded_file.getvalue()
                        # Add current message with image
                        messages_for_ollama.append({
                            'role': 'user',
                            'content': prompt_text,
                            'images': [image_bytes]
                        })
                    else:
                        # Text mode: no image
                        messages_for_ollama.append({
                            'role': 'user',
                            'content': prompt_text
                        })
                    
                    response = ollama.chat(
                        model=st.session_state.model,
                        messages=messages_for_ollama
                    )
                    full_response = response['message']['content']
                    message_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a memory error
                if "memory" in error_msg or "status code: 500" in error_msg:
                    st.error("🚫 **Memory Error: Model is too large for your system**")
                    if st.session_state.mode == "vision":
                        st.warning("""
                        **Quick Fix - Switch to Text Mode:**
                        
                        👉 **Recommended:** Switch to **Text mode** in the sidebar to use mistral/llama3 models (much lower memory)
                        
                        **If you need vision mode:**
                        
                        1. **Try a smaller Llava model:**
                           ```
                           ollama pull llava:7b
                           ```
                           Then select it from the dropdown.
                        
                        2. **Free up system memory:**
                           - Close browsers and other apps
                           - Check Task Manager for memory usage
                           - Restart if needed
                        
                        3. **Reduce image size** - Resize images to 512x512 or smaller
                        """)
                        
                        # Add a button to switch to text mode
                        if st.button("🔄 Switch to Text Mode", type="primary"):
                            st.session_state.mode = "text"
                            st.rerun()
                    else:
                        st.warning("""
                        **Try these solutions:**
                        
                        1. **Switch to a smaller model** - Try mistral:latest (smallest) instead of gpt-oss:20b
                        
                        2. **Free up system memory:**
                           - Close other applications
                           - Check Task Manager
                           - Restart if needed
                        """)
                else:
                    st.error(f"**An error occurred:** {e}")
                    st.info("💡 Make sure Ollama is running and the model is installed. Try: `ollama pull " + st.session_state.model + "`")
                
                full_response = ""  # Don't save error messages to chat history

        # Add assistant's response to history
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
