from webscraper import fetch_nvidia_financial_reports
from mistral_ocr_extractor import MistralOCRExtractor
from chunking_strategies import DocumentChunker
from vector_storage_service import VectorStorageService
from s3_utils import get_markdown_from_s3
import time

def fetch_pdf_s3_upload():
    # Step 1: Fetch NVIDIA financial reports
    print("Step 1: Fetching financial reports...")
    reports = fetch_nvidia_financial_reports()
    print("Reports fetched successfully:")
    for report in reports:
        print(f"Fetched: {report['pdf_filename']} (Size: {len(report['content'])} bytes)")
    return reports

def convert_markdown_s3_upload(reports):
    # Instantiate the OCR extractor only once
    extractor = MistralOCRExtractor()
    processed_reports = []
    
    for report in reports:
        pdf_filename = report["pdf_filename"]
        pdf_content = report["content"]
        s3_url = report["s3_url"]
        # Use a naming convention: use the part before the dot as document_id
        document_id = pdf_filename.split('.')[0]
        original_filename = pdf_filename  # Using the same filename for simplicity
        try:
            # Extract raw text and markdown content using the OCR extractor
            raw_text, markdown_content = extractor.extract_text(pdf_content, original_filename, document_id, s3_url)
            print(f"Markdown uploaded to S3: {len(markdown_content)} characters")
            
            # Add the processed report to our list
            processed_reports.append({
                "document_id": document_id,
                "pdf_filename": pdf_filename,
                "original_filename": original_filename,
                "content_length": len(markdown_content),
                "s3_url": s3_url
            })
            
        except Exception as e:
            print(f"Error converting {pdf_filename} to markdown: {e}")
    
    return processed_reports

def process_chunks_and_embeddings(processed_reports, chunking_strategy="markdown"):
    """
    Process all reports by chunking them and storing embeddings in various vector stores
    
    Args:
        processed_reports: List of processed report details
        chunking_strategy: Which chunking strategy to use
    """
    # Initialize needed services
    print(f"\nStep 3: Chunking documents using {chunking_strategy} strategy and creating embeddings...")
    chunker = DocumentChunker()
    vector_service = VectorStorageService()
    
    for report in processed_reports:
        document_id = report["document_id"]
        
        try:
            # Extract year from document_id
            year = document_id.split('_')[0]
            
            # Get markdown content from S3 using the correct path
            print(f"Retrieving markdown for {document_id} from S3...")
            markdown_filename = f"{document_id}.md"
            markdown_content = get_markdown_from_s3(document_id)
            
            # Create chunks using specified strategy
            print(f"Chunking document using {chunking_strategy} strategy...")
            chunks = chunker.chunk_document(markdown_content, strategy=chunking_strategy)
            print(f"Created {len(chunks)} chunks")
            
            # Document metadata
            metadata = {
                "document_id": document_id,
                "source_type": "pdf",
                "original_filename": report["pdf_filename"],
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunking_strategy": chunking_strategy,
                "url": report["s3_url"],
                "year": year
            }
            
            # Store document chunks in all vector databases
            print(f"Storing chunks in vector databases...")
            vector_service.store_document(document_id, chunks, metadata)
            
        except Exception as e:
            print(f"Error processing document {document_id}: {e}")

def run_pipeline(chunking_strategy="markdown"):
    """
    Run the complete NVIDIA pipeline with specified chunking strategy
    
    Args:
        chunking_strategy: Chunking strategy to use (markdown, sentence, or fixed)
    """
    print(f"Starting NVIDIA financial reports pipeline with {chunking_strategy} chunking...")
    
    # Step 1: Fetch PDFs and upload to S3
    reports = fetch_pdf_s3_upload()
    
    # Step 2: Convert PDFs to markdown and upload to S3
    processed_reports = convert_markdown_s3_upload(reports)
    
    # Step 3: Process chunks and create embeddings
    process_chunks_and_embeddings(processed_reports, chunking_strategy)
    
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    # Run the complete pipeline with markdown chunking strategy
    run_pipeline(chunking_strategy="markdown")
    
    # Uncomment to run pipeline with other chunking strategies
    # run_pipeline(chunking_strategy="sentence")
    # run_pipeline(chunking_strategy="fixed")