from app.backend.webscraper import fetch_nvidia_financial_reports
from app.backend.mistral_ocr_extractor import MistralOCRExtractor

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
    for report in reports:
        pdf_filename = report["pdf_filename"]
        pdf_content = report["content"]
        # Use a naming convention: use the part before the dot as document_id
        document_id = pdf_filename.split('.')[0]
        original_filename = pdf_filename  # Using the same filename for simplicity
        try:
            # Extract raw text and markdown content using the OCR extractor
            raw_text, markdown_content = extractor.extract_text(pdf_content, original_filename, document_id)
            print(f"Markdown uploaded to S3: {len(markdown_content)} characters")
        except Exception as e:
            print(f"Error converting {pdf_filename} to markdown: {e}")

if __name__ == '__main__':
    # Run the pipeline only once
    reports = fetch_pdf_s3_upload()
    convert_markdown_s3_upload(reports)
