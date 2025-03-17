import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import Document, DocumentResponse
from .s3_utils import list_documents_from_s3, get_document_metadata, get_markdown_from_s3

class DocumentStore:
    def __init__(self):
        """
        Initialize the document store
        No need for documents_dir as we're using S3 storage
        """
        pass
    
    def add_document(self, metadata: Dict[str, Any], content: str) -> str:
        """
        Add a document to the store
        
        Args:
            metadata: Document metadata
            content: Document content (markdown)
            
        Returns:
            Document ID
        """
        # Create document object
        document = Document(
            document_id=metadata['document_id'],
            original_filename=metadata['original_filename'],
            processing_date=metadata['processing_date'],
            content=content,
            metadata=metadata
        )
        
        # Document is already stored in S3 by the PDF processor
        # Just return the document ID
        return document.document_id
    
    def get_document_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document content by ID
        
        Args:
            document_id: Document ID
            
        Returns:
            Document content and metadata
        """
        try:
            # Get document metadata from S3
            metadata = get_document_metadata(document_id)
            if not metadata:
                return None
            
            # Get document content from S3
            filename = Path(metadata['original_filename']).stem
            content = get_markdown_from_s3(document_id, filename)
            
            return {
                "document_id": document_id,
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            print(f"Error getting document content: {str(e)}")
            return None
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the store
        
        Returns:
            List of document metadata
        """
        try:
            # Get list of document IDs from S3
            document_ids = list_documents_from_s3()
            
            # Get metadata for each document
            documents = []
            for doc_id in document_ids:
                metadata = get_document_metadata(doc_id)
                if metadata:
                    documents.append(metadata)
            
            return documents
        except Exception as e:
            print(f"Error getting documents: {str(e)}")
            return [] 