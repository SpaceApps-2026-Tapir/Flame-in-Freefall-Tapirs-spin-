#!/bin/bash
set -e

echo "=== Flame in Freefall: Starting container ==="

# Si el índice de Chroma está vacío, ejecutar la ingestión
if [ ! -d "documents/chroma/chroma.sqlite3" ] || [ -z "$(ls -A documents/chroma/ 2>/dev/null)" ]; then
    echo "=== ChromaDB index not found, ingesting documents ==="
    python scripts/ingest_documents.py
    echo "=== Ingestion complete ==="
else
    echo "=== ChromaDB index found, skipping ingestion ==="
fi

echo "=== Starting Streamlit server ==="
exec streamlit run streamlit_template.py \
    --server.port 8501 \
    --server.address 0.0.0.0
