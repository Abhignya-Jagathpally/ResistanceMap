# ResistanceMap — Multi-stage Docker build for training and inference
# Stage 1: Build environment with CUDA, Python 3.11, and dependencies
# Stage 2: Runtime with minimal footprint

# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies from requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# STAGE 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:/opt/venv/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy ResistanceMap source code
COPY . /app/

# Create directories for data, checkpoints, logs
RUN mkdir -p /app/data/raw /app/checkpoints /app/logs

# Expose ports for API and monitoring
EXPOSE 8000 6006

# Entry point — default to running the pipeline
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/h100.yaml"]
