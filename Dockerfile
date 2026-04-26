# Use a lightweight Python base image optimized for data science
FROM python:3.11-slim

# Set environment variables for non-interactive installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for PyTorch and numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Establish the working directory
WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project source tree
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Configure healthcheck for the diagnostic dashboard
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
