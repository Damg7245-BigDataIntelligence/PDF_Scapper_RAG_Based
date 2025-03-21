import os
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
from mistralai import Mistral
from dotenv import load_dotenv
from app.backend.s3_utils import upload_pdf_to_s3, AWS_S3_BUCKET_NAME, AWS_REGION

# Load environment variables
load_dotenv()

class MistralOCRExtractor:
    """Extract text from documents using Mistral's OCR API"""
    
    def __init__(self):
        """Initialize the Mistral OCR extractor"""
        # Get API key from environment
        self.api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY in your .env file.")
        
        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)
        
        # Set OCR model
        self.model = "mistral-ocr-latest"
        print("Mistral OCR extractor initialized successfully")
    
    def extract_text(self, file_content: bytes, original_filename: str, document_id: str) -> Tuple[str, str]:
        """
        Extract text from a document using Mistral OCR API
        
        Args:
            file_content: The binary content of the file
            original_filename: Original filename
            document_id: Document ID for S3 path
            
        Returns:
            Tuple of (raw_text, markdown_content)
        """
        print(f"Extracting text using Mistral OCR API for {original_filename}...")
        
        try:
            # First upload the PDF to S3 to get a URL
            # Upload the PDF to S3 and get the URL
            s3_url = upload_pdf_to_s3(file_content, original_filename, document_id)
            print(f"PDF uploaded to S3: {s3_url}")
            
            # Use the S3 URL with Mistral OCR
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": s3_url
                }
            )
            
            # Combine all pages into one markdown document
            all_pages_markdown = []
            raw_text_parts = []
            
            for page in ocr_response.pages:
                all_pages_markdown.append(page.markdown)
                # Create raw text by removing markdown formatting from the markdown text
                raw_text = page.markdown.replace('#', '').replace('*', '').replace('_', '')
                raw_text_parts.append(raw_text)
            
            markdown_content = "\n\n".join(all_pages_markdown)
            raw_text = "\n\n".join(raw_text_parts)
            
            print(f"Successfully extracted {len(markdown_content)} characters with Mistral OCR")
            
            return raw_text, markdown_content
            
        except Exception as e:
            print(f"Error processing with Mistral OCR: {str(e)}")
            raise Exception(f"Failed to process PDF with Mistral OCR: {str(e)}") 