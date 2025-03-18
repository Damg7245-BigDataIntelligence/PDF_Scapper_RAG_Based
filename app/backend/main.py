import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
from pathlib import Path

from .models import (
    Document, DocumentResponse, DocumentListResponse, 
    DocumentContentResponse, SummarizeRequest, SummarizeResponse,
    QuestionRequest, QuestionResponse, ModelsResponse,
    SearchRequest, SearchResponse, QuestionRAGRequest
)
from .pdf_processor import PDFProcessor
from .llm_service import LLMService
from .utils import DocumentStore
from .embedding_service import EmbeddingService
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Summarizer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
document_store = DocumentStore()
llm_service = LLMService()  # No API key needed for HuggingFace public models
embedding_service = EmbeddingService()

@app.get("/")
async def root():
    return {"message": "PDF Summarizer API is running"}

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available LLM models"""
    models = llm_service.get_available_models()
    return {"models": models}

@app.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """Get list of all processed documents"""
    documents = document_store.get_documents()
    return {"documents": documents}

@app.get("/documents/{document_id}", response_model=DocumentContentResponse)
async def get_document(document_id: str):
    """Get content of a specific document"""
    document_data = document_store.get_document_content(document_id)
    if not document_data:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "original_filename": document_data["metadata"]["original_filename"],
        "content": document_data["content"],
        "markdown_content": document_data["content"],
        "metadata": document_data["metadata"]
    }

@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    use_mistral: bool = Query(False, description="Use Mistral OCR instead of Docling")
):
    """Upload and process a PDF file"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read file content
    file_content = await file.read()
    
    # Create processor with selected extractor
    processor = PDFProcessor(use_mistral=use_mistral)
    
    # Process PDF
    raw_text, markdown_content, metadata = processor.process_pdf(file_content, file.filename)
    
    # Add document to store
    document_id = document_store.add_document(metadata, markdown_content)
    
    return {
        "document_id": document_id,
        "original_filename": metadata["original_filename"],
        "processing_date": metadata["processing_date"],
        "processor": "mistral_ocr" if use_mistral else "docling"
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate a summary for a document directly"""
    # Get document
    document_data = document_store.get_document_content(request.document_id)
    if not document_data:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Call LLM service directly
    summary, cost = llm_service.generate_summary(
        document_data["content"],
        request.model_id
    )
    
    return {
        "summary": summary,
        "cost": cost
    }

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks using semantic search"""
    results = embedding_service.search(request.query, request.top_k)
    return {"results": results}

@app.post("/ask_question_rag", response_model=QuestionResponse)
async def ask_question_with_rag(request: QuestionRAGRequest):
    """Answer a question using RAG (Retrieval-Augmented Generation)"""
    answer, cost = llm_service.answer_question_with_rag(
        request.question,
        request.model_id
    )
    
    return {
        "answer": answer,
        "cost": cost
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 