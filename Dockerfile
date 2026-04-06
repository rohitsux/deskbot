# DeskBot — OpenEnv Robot Desk Cleaning Environment
# Deploys to Hugging Face Spaces (port 7860, CPU-only)

FROM python:3.11-slim

# System deps for MuJoCo OSMesa headless rendering (CPU-only, no X11/GPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
# Use server-only requirements (no SB3/torch/gymnasium — keeps image small)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy full project
COPY . .

# HF Spaces runs as non-root user — ensure files are readable
RUN chmod -R 755 /app

# Expose HF Spaces default port
EXPOSE 7860

# Health check (HF Spaces pings / before marking space healthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Start FastAPI server on HF Spaces port
CMD ["uvicorn", "deskbot.server:app", "--host", "0.0.0.0", "--port", "7860"]
