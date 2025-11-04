import streamlit as st
import pdfplumber
from transformers import pipeline

# Load summarization pipeline (cached for performance)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.set_page_config(page_title="PDF Summarizer", page_icon="📄")
st.title("📄 PDF Document Summarizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if not text.strip():
        st.warning("❗ No extractable text found in the PDF.")
    else:
        st.subheader("📝 Extracted Text Preview")
        st.write(text[:1000] + "...")  # Show only first 1000 characters

        # Option to summarize
        if st.button("Summarize"):
            st.subheader("📌 Summary")
            # Hugging Face models have a limit (~1024 tokens), so split large text
            max_chunk = 1000  # characters
            chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

            summary = ""
            for chunk in chunks:
                result = summarizer(chunk, max_length=61, min_length=30, do_sample=False)
                summary += result[0]['summary_text'] + " "

            st.success(summary) 