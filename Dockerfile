FROM python:3.11-slim

# System deps some of these packages (torch, scipy, matplotlib) need to build/run cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the whole repo — this includes the gitignored trained artifacts
# (rul_model_variant_0-6.pt, scaler, dropped_idx, calibration) since Docker's
# build context is your local disk, not git.
COPY . .

EXPOSE 8501

# --server.address=0.0.0.0 is required inside a container, otherwise Streamlit
# only binds to localhost *inside* the container and the port mapping is useless.
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
