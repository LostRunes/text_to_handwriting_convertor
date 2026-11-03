# Container for Hugging Face Spaces / Render / Railway / any Docker host.
FROM python:3.11-slim

# DejaVu fonts: fallback for characters a handwriting profile is missing,
# and for the labels on the printable template.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /home/app/web

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app

ENV PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

# one process, several threads: renders are CPU-bound numpy work that releases the GIL
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300
