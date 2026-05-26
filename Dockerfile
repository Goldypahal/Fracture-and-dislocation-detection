# Use a lightweight official Python image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and NumPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model checkpoints
COPY files/ /app/files/
COPY checkpoints/ /app/checkpoints/

# Set PYTHONPATH environment variable so python can resolve files/ imports
ENV PYTHONPATH="/app/files"

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run FastAPI server using uvicorn
CMD ["uvicorn", "files.api:app", "--host", "0.0.0.0", "--port", "8000"]
