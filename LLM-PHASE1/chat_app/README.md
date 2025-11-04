# Chat App (Streamlit + FastAPI + OpenAI)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in this directory with your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
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
- Real-time chat with OpenAI's GPT models
- Simple, modern UI 