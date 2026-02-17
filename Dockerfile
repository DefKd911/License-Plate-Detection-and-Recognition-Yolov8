FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps:
# - libgl1/libglib2.0-0: common OpenCV runtime deps
# Note: EasyOCR installs via pip, no system binaries needed
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgl1 \
     libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install CPU PyTorch first (ultralytics relies on it)
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# Copy default weights (you can override with MODEL_PATH + volume mount)
COPY kbest.pt /app/kbest.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

