FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY config/ ./config/
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p data/raw data/processed models logs

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "examples/pipeline_example.py"]
