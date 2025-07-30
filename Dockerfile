# Use Python 3.10 slim image for better performance
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY webi.py main.py
COPY .env .
COPY feature_col.joblib .
COPY label_mappings.joblib .
COPY random_forest_model.joblib .

# Copy static directory and its contents
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p static temp logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port (Render will set the PORT environment variable)
EXPOSE $PORT

# Start FastAPI server
# Use $PORT environment variable that Render provides
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}