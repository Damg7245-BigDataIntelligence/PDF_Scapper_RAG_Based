# nvidia_pipeline.py

from webscraper import fetch_nvidia_financial_reports

def run_pipeline():
    # Step 1: Fetch NVIDIA financial reports
    print("Step 1: Fetching financial reports...")
    reports = fetch_nvidia_financial_reports()
    print("Reports fetched successfully:")
    for report in reports:
        print(f"Fetched: {report['pdf_filename']} (Size: {len(report['content'])} bytes)")

    # Continue with additional pipeline steps...
    # process_reports(reports)
    # analyze_data(reports)
    # etc.

if __name__ == '__main__':
    run_pipeline()
