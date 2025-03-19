import time
import requests
import streamlit as st

# API configuration
API_URL = "http://localhost:8000/"

# Set page configuration
st.set_page_config(
    page_title="PDF Summarizer and Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .stTextInput input {
        border-radius: 5px;
    }
    .stSelectbox select {
        border-radius: 5px;
    }
    .cost-info {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.5rem;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .document-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .document-preview {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .document-date {
        font-size: 0.8rem;
        color: #888;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f0f8ff;
    }
    .model-selected {
        border: 2px solid #4CAF50;
        background-color: #e8f5e9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None
if "document_content" not in st.session_state:
    st.session_state.document_content = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "cost_info" not in st.session_state:
    st.session_state.cost_info = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "huggingface/HuggingFaceH4/zephyr-7b-beta"
if "models" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/models")
        if resp.status_code == 200:
            st.session_state.models = resp.json()["models"]
        else:
            st.session_state.models = []
    except:
        st.session_state.models = []

# Helper functions
def get_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            return response.json()["documents"]
        else:
            st.error(f"Error fetching documents: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def get_document_content(document_id):
    try:
        response = requests.get(f"{API_URL}/documents/{document_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching document content: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def upload_pdf(file, use_mistral=False):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        params = {"use_mistral": str(use_mistral).lower()}
        response = requests.post(f"{API_URL}/upload_pdf", files=files, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading PDF: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def generate_summary(document_id, model_id):
    try:
        data = {"document_id": document_id, "model_id": model_id}
        response = requests.post(f"{API_URL}/summarize", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error generating summary: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def ask_question(document_id, question, model_id):
    try:
        data = {"document_id": document_id, "question": question, "model_id": model_id}
        response = requests.post(f"{API_URL}/ask_question", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error asking question: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def format_cost_info(cost_info):
    if not cost_info:
        return ""
    return f"""
    **Cost Information:**
    - Input Tokens: {cost_info['input_tokens']} (${cost_info['input_cost']:.8f})
    - Output Tokens: {cost_info['output_tokens']} (${cost_info['output_cost']:.8f})
    - Total Cost: ${cost_info['total_cost']:.8f}
    """

def search_documents(query, top_k=5):
    try:
        data = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_URL}/search", json=data)
        if response.status_code == 200:
            return response.json()["results"]
        else:
            st.error(f"Error searching documents: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def ask_question_with_rag(document_id, question, model_id="huggingface/HuggingFaceH4/zephyr-7b-beta"):
    try:
        data = {"document_id": document_id, "question": question, "model_id": model_id}
        response = requests.post(f"{API_URL}/ask_question_rag", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error asking question: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def select_model(model_id):
    st.session_state.selected_model = model_id
    st.session_state.summary = None
    st.session_state.answer = None
    st.session_state.cost_info = None

# -------------- SIDEBAR --------------
with st.sidebar:
    st.title("PDF Summarizer and Q&A")
    st.markdown("---")
    
    # RAG Selection Section
    st.subheader("RAG Selection")
    # Dropdown for RAG Option with placeholder
    rag_options = ["Select RAG Option...", "Without VectorDB", "ChromaDB", "Pinecone"]
    rag_option = st.selectbox(
        "RAG Option",
        options=rag_options,
        index=0,
        key="rag_option"
    )
    # Dropdown for Chunking Strategy with placeholder
    chunk_options = ["Select Chunking Strategy...", "Fixed-Size Chunking", "Semantic Chunking", "Sentence-Based Chunking"]
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        options=chunk_options,
        index=0,
        key="chunk_strategy"
    )
    
    st.markdown("---")
    
    # LLM Model Selection
    st.subheader("Select LLM Model")
    if not st.session_state.models:
        st.session_state.models = [{
            "id": "huggingface/HuggingFaceH4/zephyr-7b-beta",
            "name": "Zephyr 7B",
            "provider": "HuggingFace"
        }]
    for model in st.session_state.models:
        is_selected = (st.session_state.selected_model == model["id"])
        card_class = "model-card model-selected" if is_selected else "model-card"
        st.markdown(f"""
        <div class="{card_class}">
            <strong>{model["name"]}</strong><br>
            <small>Provider: {model["provider"]}</small>
        </div>
        """, unsafe_allow_html=True)
        if not is_selected:
            if st.button(f"Select {model['name']}", key=f"select_model_{model['id']}"):
                select_model(model["id"])
                try:
                    st.experimental_rerun()
                except AttributeError:
                    pass
    
    st.markdown("---")
    
    # Document Selection
    st.subheader("Document Selection")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    use_mistral = st.checkbox("Use Mistral OCR (better for scanned documents)",
                              value=False,
                              help="Select this for complex layouts or scanned documents")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                response = requests.post(
                    f"{API_URL}/upload_pdf",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    params={"use_mistral": str(use_mistral).lower()}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"PDF uploaded successfully: {data['original_filename']}")
                    processor_name = "Mistral OCR" if data.get('processor') == 'mistral_ocr' else "Docling"
                    st.info(f"Document processed using: {processor_name}")
                    time.sleep(2)
                else:
                    st.error(f"Error processing PDF: {response.text}")
    st.markdown("---")
    
    st.subheader("Select Existing Document")
    try:
        doc_resp = requests.get(f"{API_URL}/documents")
        if doc_resp.status_code == 200:
            documents = doc_resp.json()["documents"]
        else:
            documents = []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        documents = []
    if not documents:
        st.info("No processed documents found. Upload a PDF to get started.")
    else:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{doc['original_filename']}**")
                st.caption(f"Processed: {doc['processing_date']}")
            with col2:
                if st.button("Select", key=f"select_{doc['document_id']}"):
                    st.session_state.selected_document = doc
                    content_response = requests.get(f"{API_URL}/documents/{doc['document_id']}")
                    if content_response.status_code == 200:
                        st.session_state.document_content = content_response.json()
                    else:
                        st.error(f"Error fetching document content: {content_response.text}")
                    st.session_state.summary = None
                    st.session_state.answer = None
                    st.session_state.cost_info = None
                    try:
                        st.experimental_rerun()
                    except AttributeError:
                        pass

# -------------- MAIN CONTENT --------------
st.title("PDF Summarizer and Q&A")

selected_model_info = next((m for m in st.session_state.models if m["id"] == st.session_state.selected_model), None)
if selected_model_info:
    st.info(f"Using {selected_model_info['name']} by {selected_model_info['provider']}")

def generate_summary(document_id, model_id):
    try:
        data = {"document_id": document_id, "model_id": model_id}
        resp = requests.post(f"{API_URL}/summarize", json=data)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error generating summary: {resp.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def ask_question(document_id, question, model_id):
    try:
        data = {"document_id": document_id, "question": question, "model_id": model_id}
        resp = requests.post(f"{API_URL}/ask_question_rag", json=data)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error asking question: {resp.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def format_cost_info(cost_info):
    if not cost_info:
        return ""
    return f"""
    **Cost Information:**
    - Input Tokens: {cost_info['input_tokens']} (${cost_info['input_cost']:.8f})
    - Output Tokens: {cost_info['output_tokens']} (${cost_info['output_cost']:.8f})
    - Total Cost: ${cost_info['total_cost']:.8f}
    """

if st.session_state.selected_document and st.session_state.document_content:
    document = st.session_state.selected_document
    content = st.session_state.document_content
    st.header(f"Document: {document['original_filename']}")
    st.caption(f"Processed on: {document['processing_date']}")
    tab1, tab2, tab3 = st.tabs(["Document Content", "Summarize", "Ask Questions"])
    with tab1:
        st.markdown(content['markdown_content'])
    with tab2:
        if st.button("Generate Summary"):
            with st.spinner(f"Generating summary using {selected_model_info['name']}..."):
                summary_result = generate_summary(document['document_id'], st.session_state.selected_model)
                if summary_result:
                    st.session_state.summary = summary_result['summary']
                    st.session_state.cost_info = summary_result['cost']
        if st.session_state.summary:
            st.subheader("Summary")
            st.markdown(st.session_state.summary)
            if st.session_state.cost_info:
                st.markdown(format_cost_info(st.session_state.cost_info), unsafe_allow_html=True)
    with tab3:
        question = st.text_input("Ask a question about the document:")
        if question and st.button("Get Answer"):
            with st.spinner(f"Generating answer using {selected_model_info['name']}..."):
                answer_result = ask_question(document['document_id'], question, st.session_state.selected_model)
                if answer_result:
                    st.session_state.answer = answer_result['answer']
                    st.session_state.cost_info = answer_result['cost']
        if st.session_state.answer:
            st.subheader("Answer")
            st.markdown(st.session_state.answer)
            if st.session_state.cost_info:
                st.markdown(format_cost_info(st.session_state.cost_info), unsafe_allow_html=True)
else:
    st.info("Please select a document from the sidebar or upload a new PDF to get started.")
    st.markdown("""
    ## How to use this application
    
    1. **Choose your RAG Option** (Without VectorDB, ChromaDB, or Pinecone) and 
       **Pick a Chunking Strategy** (Fixed-Size, Semantic, or Sentence-Based).
    2. **Select an AI Model** (Zephyr or Gemini).
    3. **Upload a PDF** or select an existing processed document.
    4. **View the document** in the Document Content tab.
    5. **Generate a summary** in the Summarize tab.
    6. **Ask questions** in the Ask Questions tab.
    
    This application uses Large Language Models (LLMs) with optional RAG for better context retrieval.
    """)

st.markdown("---")
st.caption("PDF Summarizer and Q&A | Powered by LiteLLM, Google Gemini, and HuggingFace")
