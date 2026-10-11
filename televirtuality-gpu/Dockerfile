FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404

WORKDIR /app

# Installer les dépendances Linux
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git && \
    rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier le projet
COPY modules /app/modules
COPY models /app/models
COPY handler.py /app/handler.py

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV YOLO_MODEL_PATH=/app/models/best.pt
ENV WHISPER_MODEL=small

# Lancer le worker Runpod
CMD ["python", "-u", "handler.py"]