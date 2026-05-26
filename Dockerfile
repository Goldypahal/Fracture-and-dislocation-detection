# Use a lightweight official Python image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and NumPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Pre-install CPU-only PyTorch to avoid heavy CUDA wheels OOMing or taking forever on Hugging Face
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the remaining dependencies as the user
COPY --chown=user files/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files and model checkpoints with correct ownership
COPY --chown=user files/ $HOME/app/files/
COPY --chown=user checkpoints/ $HOME/app/checkpoints/

# Set PYTHONPATH environment variable so python can resolve files/ imports
ENV PYTHONPATH="$HOME/app/files"

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run FastAPI server using uvicorn on port 7860
CMD ["uvicorn", "files.api:app", "--host", "0.0.0.0", "--port", "7860"]
