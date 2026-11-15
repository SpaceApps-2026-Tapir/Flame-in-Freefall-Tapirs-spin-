# syntax=docker/dockerfile:1
FROM python:3.14-slim AS base

# Instalar uv para gestión de dependencias
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copiar solo los archivos de dependencias para aprovechar la caché de Docker
COPY pyproject.toml uv.lock ./

# Generar requirements.txt desde uv.lock e instalar con pip
RUN uv export --format requirements-txt --no-hashes --no-emit-project -o /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios para los documentos
RUN mkdir -p documents/pdfs documents/chroma

# Exponer el puerto de Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Script de entrada: ingesta PDFs si el índice está vacío, luego arranca Streamlit
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
