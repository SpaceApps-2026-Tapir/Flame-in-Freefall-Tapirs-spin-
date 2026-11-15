"""Small, testable security helpers shared by the app and ingestion scripts."""
from __future__ import annotations

import re
from urllib.parse import urlparse

MAX_QUESTION_CHARS = 2000
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")


def safe_http_url(value: object) -> str:
    """Return an absolute http(s) URL without embedded credentials, or an empty string.

    This is intentionally stricter than ``escape``: escaping makes a value safe inside
    HTML, but it does not stop schemes such as ``javascript:`` or ``data:``.
    """
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if (not value or len(value) > 2048 or any(char.isspace() for char in value)
            or any(char in value for char in "<>\"'")):
        return ""
    try:
        parsed = urlparse(value)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    if parsed.username or parsed.password:
        return ""
    return value


def validate_question(question: object) -> str | None:
    """Return a user-facing error when a question is too long, otherwise None."""
    text = str(question or "").strip()
    if len(text) > MAX_QUESTION_CHARS:
        return (f"Question is too long ({len(text):,} characters). "
                f"Please shorten it to {MAX_QUESTION_CHARS:,} characters or fewer.")
    return None


def safe_model_markdown(text: object) -> str:
    """Keep model text but drop Markdown links/images so the model cannot inject navigation."""
    return _MARKDOWN_LINK.sub(r"\1", str(text or ""))


def csv_safe_cell(value: object) -> object:
    """Neutralise spreadsheet formula prefixes in text cells."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_bytes(frame) -> bytes:
    """CSV download with formula-like text cells turned into plain text.

    Numeric columns are left untouched so exported measurements keep working.
    """
    safe = frame.copy()
    text_columns = safe.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        safe[column] = safe[column].map(csv_safe_cell)
    return safe.to_csv(index=False).encode("utf-8")
