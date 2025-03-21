FROM apache/airflow:2.5.1

USER root

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-airflow.txt /requirements-airflow.txt

USER airflow

# Install Python packages from requirements file
RUN pip install --no-cache-dir -r /requirements-airflow.txt