"""Module for extracting text from PDF documents using different engines"""
import io
import os
from pathlib import Path
from typing import Dict, Tuple, Any
from tempfile import NamedTemporaryFile
import logging

from s3_utils import upload_file_to_s3
from docling_core.types.doc import ImageRefMode, PictureItem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoclingExtractor:
    """Extract text from PDFs using Docling"""
    
    def __init__(self, doc_converter):
        """
        Initialize the Docling extractor
        
        Args:
            doc_converter: Initialized DocumentConverter instance
        """
        self.doc_converter = doc_converter
    
    def extract_text(self, file_content: bytes, original_filename: str, document_id: str) -> Tuple[str, str]:
        """
        Extract text from a PDF using Docling
        
        Args:
            file_content: The binary content of the PDF file
            original_filename: Original filename
            document_id: Document ID for S3 path
            
        Returns:
            Tuple of (raw_text, markdown_content)
        """
        pdf_buffer = None
        temp_file = None
        temp_file_path = None
        
        try:
            logger.info(f"Processing {original_filename} with Docling...")
            # Create a buffer from the file content
            pdf_buffer = io.BytesIO(file_content)
            
            # Create a temporary file for the PDF
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                # Write the PDF bytes to the temporary file
                pdf_buffer.seek(0)
                temp_file.write(pdf_buffer.read())
                temp_file.flush()
                temp_file_path = temp_file.name
                
                logger.info(f"Created temporary file: {temp_file_path}")
                
                # Convert the PDF file using Docling
                conv_result = self.doc_converter.convert(temp_file_path)
                logger.info("Document converted successfully")
                
                # Process the document and extract markdown
                base_name = Path(original_filename).stem
                markdown_content = self._process_document(conv_result, document_id, base_name)
                
                # Extract raw text from the document
                raw_text = self._extract_text_from_document(conv_result.document)
                
                # If we couldn't extract text, use the markdown content
                if not raw_text.strip():
                    logger.info("Using markdown content as raw text")
                    # Simple cleanup to get plain text from markdown
                    raw_text = markdown_content.replace('#', '').replace('*', '')
                
                return raw_text, markdown_content
                
        except Exception as e:
            logger.error(f"Docling processing failed: {str(e)}")
            raise Exception(f"Failed to process PDF with Docling: {str(e)}")
        finally:
            # Clean up resources
            if pdf_buffer:
                pdf_buffer.close()
            if temp_file and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {temp_file_path}: {str(e)}")
    
    def _process_document(self, conv_result, document_id: str, base_name: str) -> str:
        """
        Process the converted document and handle images if present
        
        Args:
            conv_result: The conversion result from Docling
            document_id: The document ID
            base_name: The base name of the document
            
        Returns:
            The markdown content with image references
        """
        try:
            # Try to export with PLACEHOLDER mode first
            markdown_content = conv_result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        except Exception as e:
            logger.warning(f"Error exporting with PLACEHOLDER mode: {str(e)}")
            # Try with EMBEDDED mode
            try:
                markdown_content = conv_result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            except Exception as e2:
                logger.warning(f"Error exporting with EMBEDDED mode: {str(e2)}")
                # Try without specifying image_mode
                markdown_content = conv_result.document.export_to_markdown()
        
        # Process images if they exist in the document
        picture_counter = 0
        try:
            for element, _level in conv_result.document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    
                    # Create a temporary file for the image
                    with NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                        # Save the image to the temporary file
                        element.get_image(conv_result.document).save(image_file, "PNG")
                        image_file.flush()
                        image_file_path = image_file.name
                        
                        # Define the S3 path for the image
                        image_s3_key = f"documents/images/{document_id}/{base_name}_image_{picture_counter}.png"
                        
                        # Upload the image to S3
                        with open(image_file_path, "rb") as fp:
                            image_data = fp.read()
                            image_url = upload_file_to_s3(image_data, image_s3_key, content_type="image/png")
                        
                        # Replace the image placeholder with the image URL
                        markdown_content = markdown_content.replace("<!-- image -->", f"![Image]({image_url})", 1)
                        
                        # Clean up the temporary image file
                        try:
                            os.unlink(image_file_path)
                        except Exception as e:
                            logger.warning(f"Could not delete temporary image file {image_file_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error processing images: {str(e)}")
            # Continue without images if there's an error
        
        return markdown_content
    
    def _extract_text_from_document(self, document) -> str:
        """
        Extract text from a Docling document
        
        Args:
            document: The Docling document
            
        Returns:
            The extracted text
        """
        raw_text = ""
        
        # Try different approaches to extract text
        try:
            # Try to get text directly from the document
            if hasattr(document, 'text'):
                logger.info("Using document.text")
                return document.text
            
            if hasattr(document, 'get_text_content'):
                logger.info("Using document.get_text_content()")
                return document.get_text_content()
            
            # Try to extract text from pages
            if hasattr(document, 'pages'):
                logger.info("Extracting text from pages")
                pages = document.pages
                
                # Try to iterate through pages
                try:
                    for page in pages:
                        if hasattr(page, 'text'):
                            raw_text += page.text + "\n\n"
                        elif hasattr(page, 'get_text'):
                            raw_text += page.get_text() + "\n\n"
                        elif hasattr(page, 'blocks'):
                            # Extract text from blocks
                            for block in page.blocks:
                                if hasattr(block, 'text'):
                                    raw_text += block.text + " "
                                elif hasattr(block, 'get_text'):
                                    raw_text += block.get_text() + " "
                            raw_text += "\n\n"
                except Exception as e:
                    logger.warning(f"Error iterating through pages: {str(e)}")
                    # Try to access pages by index
                    try:
                        for i in range(len(pages)):
                            page = pages[i]
                            if hasattr(page, 'text'):
                                raw_text += page.text + "\n\n"
                            elif hasattr(page, 'get_text'):
                                raw_text += page.get_text() + "\n\n"
                    except Exception as e2:
                        logger.warning(f"Error accessing pages by index: {str(e2)}")
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
        
        return raw_text


class MistralExtractor:
    """Extract text from PDFs using Mistral OCR"""
    
    def __init__(self, client, model):
        """
        Initialize the Mistral OCR extractor
        
        Args:
            client: Mistral client
            model: Mistral OCR model name
        """
        self.client = client
        self.model = model
    
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
        from .s3_utils import upload_pdf_to_s3
        
        logger.info(f"Extracting text using Mistral OCR API for {original_filename}...")
        
        try:
            # First upload the PDF to S3 to get a URL
            s3_url = upload_pdf_to_s3(file_content, original_filename, document_id)
            logger.info(f"PDF uploaded to S3: {s3_url}")
            
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
            
            logger.info(f"Successfully extracted {len(markdown_content)} characters with Mistral OCR")
            
            return raw_text, markdown_content
            
        except Exception as e:
            logger.error(f"Error processing with Mistral OCR: {str(e)}")
            raise Exception(f"Failed to process PDF with Mistral OCR: {str(e)}") 