# Fencing Analyzer — Multi-Arch Docker Image
# CPU:   docker build -t fencing-analyzer .
# GPU:   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 \
#                     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#                     -t fencing-analyzer:gpu .
# Run:   docker run --gpus all -p 8501:8501 -v /videos:/videos fencing-analyzer:gpu
#        -> open http://localhost:8501

# === BUILD ARGS ===
ARG BASE_IMAGE=python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM ${BASE_IMAGE} AS base

WORKDIR /app

# Install system deps: ffmpeg + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with correct backend (CPU or CUDA via build arg)
ARG TORCH_INDEX
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

# Copy app code
COPY app.py worker_analyze.py report_generator.py ./
COPY reports ./reports
RUN mkdir -p /app/reports

# Verify GPU availability at build time
RUN python -c "import torch,sys;c=torch.cuda.is_available();n=torch.cuda.device_count() if c else 0;print(f'PyTorch {torch.__version__} - CUDA: {c} ({n} GPU(s))');[print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(n)];sys.exit(0)"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501').read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]