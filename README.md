# LLM Projects Collection

A comprehensive collection of Large Language Model (LLM) applications showcasing different phases of development, from API-based solutions to fully local implementations.

## 📁 Project Structure

This repository contains three main phases of LLM projects:

```
LLM/
├── LLM-PHASE1/          # API-based LLM applications
│   ├── chat_app/        # OpenAI chat application
│   └── doc_summarizer/  # Document summarizer with Hugging Face
├── LLM-PHASE2/          # Hybrid local/API applications
│   ├── app.py                      # Multimodal Assistant
│   ├── meeting_extractor_local.py  # Meeting notes extractor
│   └── news_tracker.py             # News topic tracker
└── LLM-PHASE3/          # Fully local RAG application
    └── app.py           # Document Q&A with RAG
```

---

## 🚀 LLM-PHASE1: API-Based Applications

### Projects Overview

**Phase 1** focuses on cloud-based LLM services using APIs. These projects demonstrate integration with external AI services.

#### 1. Chat App (`chat_app/`)
A real-time chat application using OpenAI's GPT models.

**Features:**
- Real-time chat interface
- Integration with OpenAI API
- Modern Streamlit UI
- FastAPI backend

**Tech Stack:**
- Frontend: Streamlit
- Backend: FastAPI
- LLM: OpenAI GPT models
- API: OpenAI API

**Setup:**
```bash
cd LLM-PHASE1/chat_app
pip install -r requirements.txt
# Create .env file with OPENAI_API_KEY
uvicorn backend:app --reload  # Terminal 1
streamlit run frontend.py      # Terminal 2
```

#### 2. Document Summarizer (`doc_summarizer/`)
PDF document summarization using Hugging Face transformers.

**Features:**
- PDF upload and processing
- Automatic summarization
- Clean, user-friendly interface

**Tech Stack:**
- Frontend: Streamlit
- Backend: FastAPI
- Models: Hugging Face Transformers
- Processing: pdfplumber, PyTorch

**Setup:**
```bash
cd LLM-PHASE1/doc_summarizer
pip install -r requirements.txt
# Create .env file with HF_API_KEY
uvicorn backend:app --reload  # Terminal 1
streamlit run frontend.py      # Terminal 2
```

---

## 🎯 LLM-PHASE2: Hybrid Local/API Applications

### Projects Overview

**Phase 2** introduces local AI models using Ollama while maintaining some cloud integrations. These applications run primarily on your machine for privacy and cost efficiency.

#### 1. Multimodal Assistant (`app.py`)
A local assistant that supports both text and image inputs.

**Features:**
- Text and vision modes
- Image analysis capabilities
- Local model execution via Ollama
- Memory-efficient operation

**Tech Stack:**
- Frontend: Streamlit
- LLM: Ollama (Mistral, Llama 3, etc.)
- Image Processing: PIL/Pillow

**Setup:**
```bash
cd LLM-PHASE2
pip install -r requirements.txt
ollama pull mistral    # or llama3
streamlit run app.py
```

#### 2. Meeting Notes Extractor (`meeting_extractor_local.py`)
Extract structured notes and action items from meeting audio files.

**Features:**
- Audio transcription using Whisper
- Automated meeting summaries
- Action item extraction
- 100% local execution
- Support for MP3, WAV, M4A formats

**Tech Stack:**
- Transcription: OpenAI Whisper / faster-whisper
- LLM: Ollama (Llama 3)
- Audio Processing: ffmpeg

**Setup:**
```bash
cd LLM-PHASE2
# Install ffmpeg (required)
pip install -r requirements.txt
ollama pull llama3
streamlit run meeting_extractor_local.py
```

#### 3. News Topic Tracker (`news_tracker.py`)
Track trending topics from Google News with AI-powered summaries.

**Features:**
- Real-time news aggregation
- Topic-based news tracking
- AI-generated summaries
- Local LLM processing

**Tech Stack:**
- News Feed: Feedparser, Google News RSS
- LLM: Ollama
- Web Scraping: BeautifulSoup4

**Setup:**
```bash
cd LLM-PHASE2
pip install -r requirements.txt
ollama pull llama3  # or mistral
streamlit run news_tracker.py
```

---

## 🔒 LLM-PHASE3: Fully Local RAG Application

### Project Overview

**Phase 3** demonstrates a complete Retrieval-Augmented Generation (RAG) system running entirely on your local machine.

#### Custom Chatbot Q&A (`app.py`)
A RAG application for querying your own documents with local LLMs.

**Features:**
- 100% local and private
- No API keys required
- Document upload and processing
- Vector-based semantic search
- Context-aware Q&A

**Tech Stack:**
- Frontend: Streamlit
- LLM: Ollama (Llama 3)
- Vector Database: ChromaDB
- Framework: LangChain
- Document Processing: PyPDF

**Setup:**
```bash
cd LLM-PHASE3
pip install -r requirements.txt
ollama pull llama3
streamlit run app.py
```

**How It Works:**
1. Upload PDF documents
2. Documents are split into chunks
3. Embeddings are generated and stored in ChromaDB
4. Queries retrieve relevant context
5. Local LLM generates answers based on retrieved context

---

## 📋 Common Prerequisites

### System Requirements
- **Python 3.8+**
- **pip** (Python package manager)
- **Virtual environment** (recommended)

### Additional Tools

#### For Phase 2 & 3:
- **Ollama** - Download from [ollama.com](https://ollama.com)
  ```bash
  # After installation
  ollama pull llama3
  ollama pull mistral  # Optional
  ```

#### For Meeting Extractor:
- **ffmpeg** - Required for audio processing
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`

---

## 🔑 API Keys (Phase 1 Only)

### OpenAI API Key
Required for `LLM-PHASE1/chat_app/`:
```env
OPENAI_API_KEY=your_key_here
```

### Hugging Face API Key
Required for `LLM-PHASE1/doc_summarizer/`:
```env
HF_API_KEY=your_key_here
```

**Note:** Phase 2 and Phase 3 applications run entirely locally and do not require API keys.

---

## 🛠️ Installation Guide

### Quick Start

1. **Clone or navigate to the repository:**
   ```bash
   cd LLM
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies for each project:**
   ```bash
   # Phase 1 - Chat App
   cd LLM-PHASE1/chat_app
   pip install -r requirements.txt
   
   # Phase 1 - Document Summarizer
   cd LLM-PHASE1/doc_summarizer
   pip install -r requirements.txt
   
   # Phase 2
   cd LLM-PHASE2
   pip install -r requirements.txt
   
   # Phase 3
   cd LLM-PHASE3
   pip install -r requirements.txt
   ```

4. **Set up Ollama (for Phase 2 & 3):**
   ```bash
   # Download and install Ollama from ollama.com
   # Then pull required models:
   ollama pull llama3
   ollama pull mistral  # Optional
   ```

---

## 🎓 Learning Path

These projects are designed to demonstrate a progressive learning path:

1. **Phase 1**: Learn to integrate with cloud-based LLM APIs
2. **Phase 2**: Transition to local models with Ollama
3. **Phase 3**: Build advanced RAG applications with vector databases

---

## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection Errors (Phase 2 & 3)
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`
- Pull required models: `ollama pull llama3`

#### Memory Issues
- Use lighter models (e.g., `mistral` instead of `llama3`)
- Close other applications
- Use text-only modes when available

#### ffmpeg Not Found (Meeting Extractor)
- Install ffmpeg and ensure it's in your system PATH
- Restart your terminal after installation

#### Streamlit Not Found
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## 📝 Notes

- Each phase builds upon concepts from the previous one
- Phase 1 requires internet connection and API keys
- Phase 2 & 3 can run completely offline after initial setup
- All projects use Streamlit for the user interface
- Virtual environments are recommended to avoid dependency conflicts

---

## 🔮 Future Enhancements

- Multi-file support in RAG application
- Advanced query refinement
- Support for more document types (DOCX, TXT, etc.)
- Enhanced UI/UX improvements
- Performance optimizations

---

## 📄 License

This project collection is provided for educational and development purposes.

---

## 🤝 Contributing

Feel free to explore, modify, and extend these projects for your own use cases!

---

**Happy Coding! 🚀**

