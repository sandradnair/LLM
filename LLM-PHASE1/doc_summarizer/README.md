# Document Summarizer (Streamlit + FastAPI + Hugging Face)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in this directory with your Hugging Face API key:
   ```env
   HF_API_KEY=your_huggingface_api_key_here
   ```

## Running the App
1. Start the FastAPI backend:
   ```bash
   uvicorn backend:app --reload
   ```
2. In another terminal, run the Streamlit frontend:
   ```bash
   streamlit run frontend.py
   ```

## Features
- Upload PDF and get a summary using Hugging Face models
- Clean, easy-to-use interface 