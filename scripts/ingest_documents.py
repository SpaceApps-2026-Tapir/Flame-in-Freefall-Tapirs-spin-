"""Run with uv run python scripts/ingest_documents.py. AI-assisted implementation."""
import csv
import hashlib
import io
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vector_store import build_index
ROOT = Path(__file__).resolve().parents[1] / "documents"


def ingest():
    with (ROOT / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Add at least one document to documents/manifest.csv first.")
    seen = set()
    chunks = []
    for row in rows:
        doc_id = row.get("document_id", "")
        if not doc_id or doc_id in seen or not row.get("title"):
            raise ValueError("Each document needs a unique document_id and a title.")
        seen.add(doc_id)
        url = row.get("pdf_url", "")
        if urlparse(url).scheme != "https" or not urlparse(url).hostname:
            raise ValueError(f"{doc_id}: pdf_url must be a direct HTTPS PDF URL.")
        # Hash URLs to prevent manifest values from becoming local paths.
        cache = ROOT / "pdfs" / (hashlib.sha256(url.encode()).hexdigest() + ".pdf")
        if not cache.exists():
            if urlparse(url).hostname == "drive.google.com":
                import gdown
                file_id = parse_qs(urlparse(url).query).get("id", [""])[0]
                if not file_id:
                    raise ValueError(f"{doc_id}: Drive download URL needs a file ID.")
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".download")
                try:
                    result = gdown.download(id=file_id, output=str(temporary), use_cookies=False, timeout=60)
                    if not result:
                        raise ValueError(f"{doc_id}: Drive download failed.")
                    data = temporary.read_bytes()
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                with urlopen(Request(url, headers={"User-Agent": "FlameInFreefall/0.1"}), timeout=60) as response:
                    data = response.read(50 * 1024 * 1024 + 1)
            if len(data) > 50 * 1024 * 1024 or not data.startswith(b"%PDF-"):
                raise ValueError(f"{doc_id}: expected a PDF no larger than 50 MB.")
            PdfReader(io.BytesIO(data))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        before = len(chunks)
        digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        for page_no, page in enumerate(PdfReader(cache).pages, 1):
            words = (page.extract_text() or "").split()
            # Short passages reduce truncation by the MiniLM embedding model.
            for start in range(0, len(words), 80):
                text = " ".join(words[start:start + 100])
                if text:
                    chunks.append({"document_id": doc_id, "title": row["title"],
                                   "experiment": row.get("experiment", ""), "pdf_url": url,
                                   "source_page_url": row.get("source_page_url", url),
                                   "page": page_no, "sha256": digest, "text": text})
        if len(chunks) == before:
            raise ValueError(f"{doc_id}: no text extracted; scanned PDFs need OCR first.")
        print(f"Indexed {doc_id}: {len(chunks) - before} passages")
    build_index(chunks)
    print(f"Published Chroma index: {len(chunks)} passages")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from None
