import os
import io
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import tempfile
from tempfile import NamedTemporaryFile
from .s3_utils import upload_pdf_to_s3, upload_markdown_to_s3, upload_file_to_s3

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

class PDFProcessor:
    def __init__(self):
        """
        Initialize the PDF processor with Docling configuration
        """
        # Initialize Docling document converter with appropriate options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = False  # Set to True if you want page images
        pipeline_options.generate_picture_images = False  # Set to True if you want picture images
        
        self.doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        print("Docling initialized successfully")
    
    def process_pdf(self, file_content: bytes, original_filename: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process a PDF file using Docling and extract its text content, storing in S3
        
        Args:
            file_content: The binary content of the PDF file
            original_filename: The original filename of the PDF
            
        Returns:
            Tuple containing the raw text content, markdown formatted content, and metadata
        """
        pdf_buffer = None
        temp_file = None
        
        try:
            print("Processing with Docling...")
            # Create a buffer from the file content
            pdf_buffer = io.BytesIO(file_content)
            
            # Get base name for file naming
            base_name = Path(original_filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            document_id = f"{base_name}_{timestamp}"
            
            # Create a temporary file for the PDF
            with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                # Write the PDF bytes to the temporary file
                pdf_buffer.seek(0)
                temp_file.write(pdf_buffer.read())
                temp_file.flush()
                temp_file_path = temp_file.name
                
                print(f"Created temporary file: {temp_file_path}")
                
                # Convert the PDF file using Docling
                conv_result = self.doc_converter.convert(temp_file_path)
                print("Document converted successfully")
                
                # Process the document and extract markdown
                markdown_content = self._process_document(conv_result, document_id, base_name)
                
                # Extract raw text from the document
                raw_text = self._extract_text_from_document(conv_result.document)
                
                # If we couldn't extract text, use the markdown content
                if not raw_text.strip():
                    print("Using markdown content as raw text")
                    # Simple cleanup to get plain text from markdown
                    raw_text = markdown_content.replace('#', '').replace('*', '')
                
                # Upload PDF to S3
                pdf_url = upload_pdf_to_s3(file_content, original_filename, document_id)
                
                # Upload markdown to S3
                markdown_url = upload_markdown_to_s3(markdown_content, document_id, base_name)
                
                metadata = {
                    'document_id': document_id,
                    'source_type': 'pdf',
                    'original_filename': original_filename,
                    'processing_date': timestamp,
                    'content_type': 'document',
                    'pdf_url': pdf_url,
                    'markdown_url': markdown_url,
                    'processor': 'docling'
                }
                
                print("Docling processing successful")
                return raw_text, markdown_content, metadata
                
        except Exception as e:
            print(f"Docling processing failed: {str(e)}")
            raise Exception(f"Failed to process PDF with Docling: {str(e)}")
        finally:
            # Clean up resources
            if pdf_buffer:
                pdf_buffer.close()
            if temp_file and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file_path}: {str(e)}")
    
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
            print(f"Error exporting with PLACEHOLDER mode: {str(e)}")
            # Try with EMBEDDED mode
            try:
                markdown_content = conv_result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            except Exception as e2:
                print(f"Error exporting with EMBEDDED mode: {str(e2)}")
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
                            print(f"Warning: Could not delete temporary image file {image_file_path}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error processing images: {str(e)}")
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
                print("Using document.text")
                return document.text
            
            if hasattr(document, 'get_text_content'):
                print("Using document.get_text_content()")
                return document.get_text_content()
            
            # Try to extract text from pages
            if hasattr(document, 'pages'):
                print("Extracting text from pages")
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
                    print(f"Error iterating through pages: {str(e)}")
                    # Try to access pages by index
                    try:
                        for i in range(len(pages)):
                            page = pages[i]
                            if hasattr(page, 'text'):
                                raw_text += page.text + "\n\n"
                            elif hasattr(page, 'get_text'):
                                raw_text += page.get_text() + "\n\n"
                    except Exception as e2:
                        print(f"Error accessing pages by index: {str(e2)}")
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
        
        return raw_text