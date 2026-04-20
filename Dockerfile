# STEP 1: Use an official Python runtime as a parent image
# We use 'slim' to keep the image size small and professional
FROM python:3.10-slim

# STEP 2: Set the working directory inside the container
WORKDIR /app

# STEP 3: Install system-level dependencies for Geospatial libraries
# Libraries like 'rasterio' need these C-based tools to handle .tif files
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# STEP 4: Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Copy your project files into the container
# This includes your models/, data/, and scripts/
COPY . .

# STEP 6: Expose the port Streamlit runs on
EXPOSE 8501

# STEP 7: Define the command to run your app
# Using '0.0.0.0' allows the container to be accessed from your local browser
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]