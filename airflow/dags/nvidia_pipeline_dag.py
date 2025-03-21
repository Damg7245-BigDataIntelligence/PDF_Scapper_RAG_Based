from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys

# Add the path to your backend directory
sys.path.append('/opt/airflow/app/backend')

# Import functions from nvidia_pipeline
from nvidia_pipeline import (
    scrape_annual_reports,
    extract_text_from_pdfs,
    process_and_chunk_text,
    create_embeddings,
    upload_to_vector_db
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'nvidia_financial_pipeline',
    default_args=default_args,
    description='Process NVIDIA financial reports',
    schedule_interval=timedelta(days=7),  # Weekly
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['nvidia', 'financial', 'pdf', 'embeddings'],
)

# Define tasks
start = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

scrape_reports_task = PythonOperator(
    task_id='scrape_annual_reports',
    python_callable=scrape_annual_reports,
    op_kwargs={
        'url': 'https://investor.nvidia.com/financial-info/annual-reports-and-proxy/default.aspx',
        'download_dir': '/opt/airflow/app/backend/data/raw_pdfs'
    },
    dag=dag,
)

extract_text_task = PythonOperator(
    task_id='extract_text_from_pdfs',
    python_callable=extract_text_from_pdfs,
    op_kwargs={
        'pdf_dir': '/opt/airflow/app/backend/data/raw_pdfs',
        'output_dir': '/opt/airflow/app/backend/data/extracted_text'
    },
    dag=dag,
)

chunk_text_task = PythonOperator(
    task_id='process_and_chunk_text',
    python_callable=process_and_chunk_text,
    op_kwargs={
        'text_dir': '/opt/airflow/app/backend/data/extracted_text',
        'output_dir': '/opt/airflow/app/backend/data/chunks'
    },
    dag=dag,
)

create_embeddings_task = PythonOperator(
    task_id='create_embeddings',
    python_callable=create_embeddings,
    op_kwargs={
        'chunks_dir': '/opt/airflow/app/backend/data/chunks',
        'embeddings_file': '/opt/airflow/app/backend/data/embeddings/nvidia_embeddings.json'
    },
    dag=dag,
)

upload_to_db_task = PythonOperator(
    task_id='upload_to_vector_db',
    python_callable=upload_to_vector_db,
    op_kwargs={
        'embeddings_file': '/opt/airflow/app/backend/data/embeddings/nvidia_embeddings.json',
        'collection_name': 'nvidia_financials'
    },
    dag=dag,
)

end = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define dependencies
start >> scrape_reports_task >> extract_text_task >> chunk_text_task >> create_embeddings_task >> upload_to_db_task >> end