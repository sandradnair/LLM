# 📚 Custom Chatbot Q&A with Local LLMs (RAG Application)

This project is a **fully local, private Retrieval-Augmented Generation (RAG) application**.  
It allows you to upload your own documents and ask questions about them using a **Large Language Model (LLM) running entirely on your machine**.  

✅ **100% Local & Private** – No data ever leaves your system  
✅ **No API Keys Needed** – Uses open-source models via [Ollama](https://ollama.com)  
✅ **Simple UI** – Built with Streamlit for an easy user experience  
✅ **PDF Document Support** – Upload and query your PDFs  
✅ **Docker-Free Setup** – Simple installation without containers  

---

## ⚡ Core Technologies
- **Frontend (UI):** Streamlit  
- **LLM Serving:** Ollama (models like Llama 3, Mistral, etc.)  
- **Backend Orchestration:** LangChain  
- **Vector Database:** ChromaDB  
- **Document Loading:** PyPDFLoader  

---

## 🔧 Local Setup and Installation

### 1. Prerequisites
- Python **3.8+**
- [Ollama](https://ollama.com) installed and running

---

### 2. Install Ollama & Download a Model

#### Install Ollama
- **macOS/Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
Windows:
Download and run the installer from Ollama Downloads.

Pull a Model (recommended: llama3)
bash
Copy code
ollama pull llama3
ℹ️ Ollama starts a background server automatically after installation.

3. Set Up the Project
Clone the repository:

bash
Copy code
git clone <your-repo-url>
cd custom-rag-app
Or manually create a folder custom-rag-app/ and add app.py + requirements.txt.

Create a Virtual Environment
bash
Copy code
python -m venv venv
Activate the environment:

macOS/Linux:

bash
Copy code
source venv/bin/activate
Windows:

bash
Copy code
venv\Scripts\activate
Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the Application
bash
Copy code
streamlit run app.py
Your browser will open at 👉 http://localhost:8501

🚀 How to Use
Upload a PDF Document – Select a file from the sidebar.

Process the Document – Click the button to:

Load & split the PDF

Generate embeddings

Store them in ChromaDB

Ask a Question – Type your query in the input box.

Get an Answer – The app retrieves relevant chunks & generates an answer using your local LLM.

📂 Project Structure
bash
Copy code
custom-rag-app/
│
├── venv/              # Python virtual environment
├── app.py             # Main Streamlit app
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
🛠 Troubleshooting
Ollama Connection Error
Ensure Ollama is running (desktop app open or ollama serve in a terminal).

Slow Performance
Try a lighter model (e.g., mistral) if response time is slow. Performance depends on your CPU/GPU.

streamlit command not found
Activate your virtual environment first:

bash
Copy code
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
🎯 Future Enhancements
Multi-file support

Advanced query refinement

Support for more document types (DOCX, TXT, etc.)