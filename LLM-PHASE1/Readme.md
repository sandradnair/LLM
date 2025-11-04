# LLM Phase 1: API-Based Applications

Welcome to **LLM Phase 1**, which focuses on building LLM applications using cloud-based APIs. This phase introduces you to integrating with external AI services and demonstrates how to structure applications with separate frontend and backend components.

## 📋 Overview

Phase 1 contains two projects that showcase different approaches to using LLM APIs:

1. **Chat App** - A real-time chat application using OpenAI's GPT models with a FastAPI backend
2. **Document Summarizer** - A PDF document summarization tool using Hugging Face transformers

Both projects demonstrate:
- Integration with cloud-based LLM services
- Modern web application architecture
- User-friendly Streamlit interfaces
- API key management and environment variables

---

## 🗂️ Project Structure

```
LLM-PHASE1/
├── chat_app/
│   ├── backend.py          # FastAPI backend server
│   ├── frontend.py         # Streamlit frontend interface
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Chat app specific documentation
├── doc_summarizer/
│   ├── frontend.py         # Streamlit application (standalone)
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Document summarizer specific documentation
└── venv/                   # Virtual environment (shared)
```

---

## 🚀 Project 1: Chat App

### Description
A real-time chat application that allows users to have conversations with OpenAI's GPT-3.5-turbo model. The application features a clean, modern interface built with Streamlit and a robust FastAPI backend.

### Features
- 💬 Real-time chat interface
- 🔄 Conversation history management
- 🎨 Modern, user-friendly UI
- ⚡ FastAPI backend for API handling
- 🔒 Secure API key management via environment variables

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **LLM**: OpenAI GPT-3.5-turbo
- **API**: OpenAI API
- **Dependencies**: `openai`, `fastapi`, `uvicorn`, `streamlit`, `python-dotenv`

### Prerequisites
- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation & Setup

1. **Navigate to the chat_app directory:**
   ```bash
   cd LLM-PHASE1/chat_app
   ```

2. **Create a virtual environment (if not already created):**
   ```bash
   python -m venv ../venv
   
   # Activate on Windows:
   ../venv\Scripts\activate
   
   # Activate on macOS/Linux:
   source ../venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the `chat_app` directory:**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   **Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Running the Application

The application requires two terminals - one for the backend and one for the frontend.

**Terminal 1 - Start the FastAPI backend:**
```bash
cd LLM-PHASE1/chat_app
# Activate virtual environment first
uvicorn backend:app --reload
```
The backend will start on `http://localhost:8000`

**Terminal 2 - Start the Streamlit frontend:**
```bash
cd LLM-PHASE1/chat_app
# Activate virtual environment first
streamlit run frontend.py
```
The frontend will open in your browser at `http://localhost:8501`

### Usage
1. Open the Streamlit app in your browser
2. Type your message in the input field
3. Click "Send" to get a response from GPT-3.5-turbo
4. Your conversation history is maintained throughout the session

### API Endpoints

The FastAPI backend provides the following endpoint:

- **POST `/chat`**
  - Request body:
    ```json
    {
      "message": "Your message here",
      "history": [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"}
      ]
    }
    ```
  - Response:
    ```json
    {
      "reply": "AI response text"
    }
    ```

---

## 📄 Project 2: Document Summarizer

### Description
A PDF document summarization tool that uses Hugging Face's BART model to generate concise summaries of uploaded PDF documents. This is a standalone Streamlit application that runs entirely in the frontend.

### Features
- 📤 PDF file upload
- 📝 Text extraction from PDFs
- 🤖 AI-powered summarization using BART model
- 👁️ Text preview before summarization
- ⚡ Fast and efficient processing

### Tech Stack
- **Frontend**: Streamlit (standalone)
- **LLM**: Hugging Face BART (facebook/bart-large-cnn)
- **PDF Processing**: pdfplumber
- **ML Framework**: PyTorch, Transformers
- **Dependencies**: `streamlit`, `pdfplumber`, `transformers`, `torch`

### Prerequisites
- Python 3.8+
- Sufficient RAM (for loading the BART model)
- Internet connection (for downloading the model on first run)

### Installation & Setup

1. **Navigate to the doc_summarizer directory:**
   ```bash
   cd LLM-PHASE1/doc_summarizer
   ```

2. **Create a virtual environment (if not already created):**
   ```bash
   python -m venv ../venv
   
   # Activate on Windows:
   ../venv\Scripts\activate
   
   # Activate on macOS/Linux:
   source ../venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The first time you run the app, it will download the BART model (~1.6GB), which may take a few minutes.

### Running the Application

**Single command to start:**
```bash
cd LLM-PHASE1/doc_summarizer
# Activate virtual environment first
streamlit run frontend.py
```

The application will open in your browser at `http://localhost:8501`

### Usage
1. Open the Streamlit app in your browser
2. Click "Upload a PDF file" and select your PDF document
3. Wait for the text extraction to complete
4. Review the extracted text preview (first 1000 characters)
5. Click "Generate Summary" to create a summary
6. View the generated summary

### Model Information
- **Model**: `facebook/bart-large-cnn`
- **Type**: Abstractive summarization
- **Size**: ~1.6GB
- **First Run**: Model will be downloaded automatically from Hugging Face

---

## 🔑 API Keys & Environment Variables

### Chat App
Requires an OpenAI API key stored in a `.env` file:

```env
OPENAI_API_KEY=sk-...
```

**Getting an OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy and paste it into your `.env` file

**Important**: Never commit your `.env` file to version control!

### Document Summarizer
No API key required! The application uses open-source models from Hugging Face that run locally.

---

## 🛠️ Common Setup Issues & Solutions

### Issue: ModuleNotFoundError
**Solution**: Ensure you've activated the virtual environment and installed all requirements:
```bash
# Activate venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Issue: OpenAI API Key Error
**Solution**: 
- Verify your `.env` file exists in the `chat_app` directory
- Check that the API key is correct and active
- Ensure `python-dotenv` is installed

### Issue: Port Already in Use
**Solution**: 
- FastAPI default port: 8000 - Change with `uvicorn backend:app --reload --port 8001`
- Streamlit default port: 8501 - Change with `streamlit run frontend.py --server.port 8502`

### Issue: Out of Memory (Document Summarizer)
**Solution**: 
- Close other applications
- Use a smaller model or reduce batch size
- Ensure you have at least 4GB RAM available

### Issue: Model Download Fails
**Solution**:
- Check your internet connection
- Ensure you have sufficient disk space (~2GB)
- Try running the app again (download will resume)

---

## 📚 Learning Objectives

By completing Phase 1, you will have learned:

1. ✅ How to integrate OpenAI API into applications
2. ✅ How to structure FastAPI backends for LLM applications
3. ✅ How to build user interfaces with Streamlit
4. ✅ How to use Hugging Face transformers for NLP tasks
5. ✅ How to handle PDF processing and text extraction
6. ✅ How to manage API keys securely with environment variables
7. ✅ How to implement conversation history in chat applications

---

## 🔄 Next Steps

After completing Phase 1, you can move on to:

- **LLM-PHASE2**: Hybrid local/API applications using Ollama
- **LLM-PHASE3**: Fully local RAG applications with vector databases

---

## 📝 Notes

- Both projects can share the same virtual environment
- The chat app requires an active internet connection
- The document summarizer downloads models on first run but can work offline afterward
- API costs may apply for OpenAI usage (check [OpenAI pricing](https://openai.com/pricing))

---

## 🤝 Contributing

Feel free to modify and extend these projects:
- Add support for more file formats
- Implement additional features
- Improve UI/UX
- Add error handling and validation

---

## 📄 License

These projects are provided for educational purposes.

---

**Happy Learning! 🚀**

