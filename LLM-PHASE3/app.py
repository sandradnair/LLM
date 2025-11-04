import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- App Configuration ---
st.set_page_config(page_title="Local RAG Q&A with Ollama", layout="wide")
st.title("Custom Chatbot Q&A (RAG Application)")
st.markdown("""
This application allows you to ask questions about your documents. 
It uses local models through Ollama, so your data never leaves your machine.
- **Backend**: LangChain, Ollama (`llama3`)
- **Database**: ChromaDB (for vector storage)
- **UI**: Streamlit
""")

# --- Core Components Initialization ---
# Initialize Ollama embeddings and the LLM
# Ensure you have a model like 'llama3' pulled via `ollama pull llama3`
try:
    ollama_embeddings = OllamaEmbeddings(model="llama3")
    ollama_llm = Ollama(model="llama3")
except Exception as e:
    st.error(f"Failed to initialize Ollama. Please ensure Ollama is running and the model is available. Error: {e}")
    st.stop()

# Use session state to persist the RAG chain across reruns
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# --- UI Sidebar for Document Upload and Processing ---
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document... This may take a moment."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 1. Load the document
                loader = PyPDFLoader(tmp_file_path)
                data = loader.load()

                # 2. Split the document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                all_splits = text_splitter.split_documents(data)

                # 3. Create and store embeddings in ChromaDB
                vectorstore = Chroma.from_documents(
                    documents=all_splits, 
                    embedding=ollama_embeddings
                )

                # 4. Create the retrieval chain using LangChain Expression Language (LCEL)
                retriever = vectorstore.as_retriever()
                
                # Define the prompt template
                template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                # Format documents function
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # Create the RAG chain using LCEL
                # This chain structure: question -> retrieve docs -> format -> prompt -> LLM -> parse
                def create_rag_chain_input(question):
                    docs = retriever.invoke(question)
                    return {"context": format_docs(docs), "question": question}
                
                st.session_state.rag_chain = (
                    RunnablePassthrough() 
                    | RunnableLambda(create_rag_chain_input) 
                    | prompt 
                    | ollama_llm 
                    | StrOutputParser()
                )
                
                # Store the retriever for later use
                st.session_state.retriever = retriever

                st.success("Document processed! You can now ask questions.")

                # Clean up the temporary file
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"An error occurred during document processing: {e}")

# --- Main Q&A Interface ---
st.header("2. Ask a Question")

if st.session_state.rag_chain is None:
    st.warning("Please upload and process a document first using the sidebar.")
else:
    question = st.text_input("Enter your question about the document:")

    if question:
        with st.spinner("Thinking..."):
            try:
                # Invoke the RAG chain to get an answer
                response = st.session_state.rag_chain.invoke(question)
                
                # Display the answer
                st.subheader("Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred while fetching the answer: {e}")

