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

# Install project dependencies with uv sync (including cuda extras for k2)
RUN uv sync --extra cuda

# Add virtual environment to PATH
ENV PATH="/workspace/.venv/bin:$PATH"

# Default command
CMD ["bash"]
