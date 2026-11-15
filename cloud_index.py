"""Initialize the cloud's local RAG index once, across Streamlit sessions."""
from threading import Lock

from vector_store import STORE

_BUILD_LOCK = Lock()


def ensure_index(progress):
    """Build from the committed manifest; failed builds can be retried."""
    with _BUILD_LOCK:
        if (STORE / "active.json").exists() and (STORE / "chroma.sqlite3").exists():
            return False
        from scripts.ingest_documents import ingest
        progress("Downloading PDFs and preparing the document index. This can take several minutes…")
        ingest(progress=progress)
        if not (STORE / "active.json").exists():
            raise RuntimeError("Ingestion finished without publishing a document index.")
        return True
