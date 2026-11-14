"""Discover public Drive PDFs and replace the manifest (no PDF downloads)."""
import csv
import re
from pathlib import Path

import gdown

FOLDER = "https://drive.google.com/drive/folders/1YXhHQEK82FCtzmncAq9-JR7_xbUgDaXv"
ROOT = Path(__file__).resolve().parents[1] / "documents"


def sync():
    files = gdown.download_folder(url=FOLDER, skip_download=True, use_cookies=False, timeout=30)
    rows = []
    for file in files or []:
        if not file.path.lower().endswith(".pdf"):
            continue
        name = Path(file.path).stem
        match = re.search(r"\b(BASS-II|BASS|SAME-R|SAME|FLEX)\b", name.replace("_", " ").upper())
        rows.append(dict(document_id="drive-" + file.id, title=name.replace("_", " "),
                         experiment=match.group(1) if match else "",
                         pdf_url="https://drive.google.com/uc?export=download&id=" + file.id,
                         source_page_url=f"https://drive.google.com/file/d/{file.id}/view"))
    if not rows:
        raise ValueError("No PDFs found. Check folder sharing and download permissions.")
    temporary = ROOT / "manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(ROOT / "manifest.csv")
    print(f"Manifest updated: {len(rows)} Drive PDFs")


if __name__ == "__main__":
    sync()
