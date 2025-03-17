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
    # Get available models
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            st.session_state.models = response.json()["models"]
        else:
            st.session_state.models = []
    except:
        st.session_state.models = []

# Helper functions
def get_documents():
    """Get list of processed documents from API"""
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
    """Get content of a specific document from API"""
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

def upload_pdf(file):
    """Upload PDF file to API"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload_pdf", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading PDF: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def generate_summary(document_id, model_id):
    """Generate summary for a document"""
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
    """Ask a question about a document"""
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
    """Format cost information for display"""
    if not cost_info:
        return ""
    
    return f"""
    **Cost Information:**
    - Input Tokens: {cost_info['input_tokens']} (${cost_info['input_cost']:.8f})
    - Output Tokens: {cost_info['output_tokens']} (${cost_info['output_cost']:.8f})
    - Total Cost: ${cost_info['total_cost']:.8f}
    """

def select_model(model_id):
    """Select a model and update session state"""
    st.session_state.selected_model = model_id
    st.session_state.summary = None
    st.session_state.answer = None
    st.session_state.cost_info = None

# Sidebar
with st.sidebar:
    st.title("PDF Summarizer and Q&A")
    st.markdown("---")
    
    # Model selection
    st.subheader("Select LLM Model")
    
    # Get available models from API or use default if API call fails
    if not st.session_state.models:
        st.session_state.models = [
            {
                "id": "huggingface/HuggingFaceH4/zephyr-7b-beta",
                "name": "Zephyr 7B",
                "provider": "HuggingFace"
            }
        ]
    
    # Display model selection cards
    for model in st.session_state.models:
        # Determine if this model is selected
        is_selected = st.session_state.selected_model == model["id"]
        
        # Create a card with conditional styling
        card_class = "model-card model-selected" if is_selected else "model-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <strong>{model["name"]}</strong><br>
            <small>Provider: {model["provider"]}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Add select button if not already selected
        if not is_selected:
            if st.button(f"Select {model['name']}", key=f"select_model_{model['id']}"):
                select_model(model["id"])
                st.rerun()
    
    st.markdown("---")
    
    # Document selection
    st.subheader("Document Selection")
    
    # Option to upload new PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                result = upload_pdf(uploaded_file)
                if result:
                    st.success(f"PDF uploaded successfully: {result['original_filename']}")
                    # Refresh document list
                    time.sleep(2)  # Wait for processing to complete
    
    st.markdown("---")
    
    # Select from existing documents
    st.subheader("Select Existing Document")
    documents = get_documents()
    
    if not documents:
        st.info("No processed documents found. Upload a PDF to get started.")
    else:
        # Create document selection
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{doc['original_filename']}**")
                st.caption(f"Processed: {doc['processing_date']}")
            with col2:
                # Add a key parameter to the button to make it unique
                button_key = f"select_{doc['document_id']}"
                if st.button("Select", key=button_key):
                    # Check if we're already viewing this document to avoid reloading
                    if (st.session_state.selected_document is None or 
                        st.session_state.selected_document['document_id'] != doc['document_id']):
                        st.session_state.selected_document = doc
                        st.session_state.document_content = get_document_content(doc['document_id'])
                        st.session_state.summary = None
                        st.session_state.answer = None
                        st.session_state.cost_info = None
                        st.rerun()

# Main content
st.title("PDF Summarizer and Q&A")

# Display currently selected model
selected_model_info = next((m for m in st.session_state.models if m["id"] == st.session_state.selected_model), None)
if selected_model_info:
    st.info(f"Using {selected_model_info['name']} by {selected_model_info['provider']}")

if st.session_state.selected_document and st.session_state.document_content:
    document = st.session_state.selected_document
    content = st.session_state.document_content
    
    # Document info
    st.header(f"Document: {document['original_filename']}")
    st.caption(f"Processed on: {document['processing_date']}")
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Document Content", "Summarize", "Ask Questions"])
    
    # Document Content Tab
    with tab1:
        st.markdown(content['markdown_content'])
    
    # Summarize Tab
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
    
    # Ask Questions Tab
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
    # No document selected
    st.info("Please select a document from the sidebar or upload a new PDF to get started.")
    
    # Display some instructions
    st.markdown("""
    ## How to use this application
    
    1. **Select an AI model** from the sidebar (Zephyr or Gemini)
    2. **Upload a PDF document** using the file uploader in the sidebar
    3. **Select a document** from the list of processed documents
    4. **View the document content** in the Document Content tab
    5. **Generate a summary** of the document in the Summarize tab
    6. **Ask questions** about the document in the Ask Questions tab
    
    This application uses Large Language Models (LLMs) to analyze PDF documents and provide summaries and answers to your questions.
    """)

# Footer
st.markdown("---")
st.caption("PDF Summarizer and Q&A | Powered by LiteLLM, Google Gemini, and HuggingFace")