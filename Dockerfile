# Dockerfile for ZipVoice Japanese training with k2 support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install k2 with CUDA support
RUN pip install --no-cache-dir \
    k2==1.24.4.dev20241030+cuda12.4.torch2.5.1 \
    -f https://k2-fsa.github.io/k2/cuda.html

# Install piper-phonemize
RUN pip install --no-cache-dir \
    piper-phonemize \
    --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html

# Copy requirements and install dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Install pyopenjtalk-plus for Japanese support
RUN pip install --no-cache-dir pyopenjtalk-plus

# Install additional dependencies
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    lhotse \
    vocos

# Copy the project
COPY . /workspace/

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Default command
CMD ["bash"]
