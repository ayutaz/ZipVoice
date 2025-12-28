# Dockerfile for ZipVoice Japanese training with k2 support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml /workspace/
COPY uv.lock /workspace/
COPY README.md /workspace/
COPY zipvoice /workspace/zipvoice

# Install project dependencies with uv sync
RUN uv sync --frozen

# Install k2 with CUDA support (not available via pypi, need extra index)
# Using --frozen to skip resolution for other Python versions
RUN uv add --frozen k2==1.24.4.dev20241030+cuda12.4.torch2.5.1 \
    --extra-index-url https://k2-fsa.github.io/k2/cuda.html

# Install piper-phonemize (also needs extra index)
RUN uv add --frozen piper-phonemize \
    --extra-index-url https://k2-fsa.github.io/icefall/piper_phonemize.html

# pyopenjtalk-plus is already installed via uv sync

# Default command
CMD ["bash"]
