# Fencing Analyzer — Docker Image
# YOLOv8m Pose-Estimation + Streamlit UI + Media Server
#
# Build:   docker build -t fencing-analyzer .
# Run:     docker run -p 8501:8501 -v /path/to/videos:/videos fencing-analyzer
#          -> open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# Install system deps: OpenCV, ffmpeg, numpy build deps
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

# Copy app code
COPY app.py worker_analyze.py report_generator.py ./
COPY reports ./reports

# Create reports directory (writable)
RUN mkdir -p /app/reports

# Expose Streamlit + media server ports
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

# Run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
