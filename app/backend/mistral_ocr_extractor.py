import os
from typing import Tuple
from mistralai import Mistral
from dotenv import load_dotenv
from s3_utils import upload_pdf_to_s3, upload_markdown_to_s3
import base64
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
    
    def extract_text(self, file_content: bytes, original_filename: str, document_id: str, s3_url: str = None) -> Tuple[str, str]:
        """
        Extract text from a document using Mistral OCR API
        
        Args:
            file_content: The binary content of the file
            original_filename: Original filename
            document_id: Document ID for S3 path
            s3_url: Optional S3 URL of the already uploaded file
            
        Returns:
            Tuple of (raw_text, markdown_content)
        """
        print(f"Extracting text using Mistral OCR API for {original_filename}...")
        
        try:
            # Use provided S3 URL if available
            if s3_url:
                print(f"Using existing S3 URL: {s3_url}")
            else:
                # First upload the PDF to S3 to get a URL
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
            
            # Get the year from document_id (assuming format like "2023_Q1")
            year = document_id.split('_')[0]
            
            try:
                # Use upload_markdown_to_s3 with the correct path structure
                markdown_filename = f"{document_id}.md"
                markdown_url = upload_markdown_to_s3(markdown_content, year, markdown_filename)
                print(f"Markdown content uploaded to S3: {markdown_url}")
            except Exception as e:
                print(f"Warning: Failed to upload markdown to S3: {e}")
            
            return raw_text, markdown_content
            
        except Exception as e:
            print(f"Error processing with Mistral OCR: {str(e)}")
            raise Exception(f"Failed to process PDF with Mistral OCR: {str(e)}") 