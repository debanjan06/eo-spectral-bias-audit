# Use a lightweight Python base image optimized for data science
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project source tree
COPY . .

# Ensure the models directory exists
RUN mkdir -p models

# DOWNLOAD STEP: Must happen BEFORE the Entrypoint
# We only run this if the weights aren't already part of the COPY . . 
RUN curl -L -o models/best_baseline_model.pth \
    https://github.com/debanjan06/eo-spectral-bias-audit/releases/download/v1.0/best_baseline_model.pth

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the application (This must be the final instruction)
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
