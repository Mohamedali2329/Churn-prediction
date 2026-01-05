# Use Python 3.9 slim image as base
# Platform is specified via --platform flag in docker build command
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including MySQL client
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    libabsl-dev \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY code/ ./code/

# Copy models directory
COPY models/ ./models/

# Copy data directory (needed for Streamlit app)
COPY data/ ./data/

# Copy environment configuration directory
COPY environment/ ./environment/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
# 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Default command - can be overridden
# To run FastAPI: docker run -p 8000:8000 <image> uvicorn code.app.main:app --host 0.0.0.0 --port 8000
# To run Streamlit: docker run -p 8501:8501 <image> streamlit run code/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
CMD ["streamlit", "run", "code/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

