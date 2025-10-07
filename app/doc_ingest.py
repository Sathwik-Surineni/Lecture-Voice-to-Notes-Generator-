# app/doc_ingest.py
from __future__ import annotations
from typing import List, Dict
from pypdf import PdfReader
import io, re

def extract_text_from_pdf(path_or_bytes) -> str:
    """
    Extracts readable text from a non-scanned PDF.
    `path_or_bytes` can be a filesystem path or bytes-like object.
    """
    if isinstance(path_or_bytes, (bytes, bytearray, io.BytesIO)):
        reader = PdfReader(io.BytesIO(path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes.getvalue()))
    else:
        reader = PdfReader(path_or_bytes)

    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t.strip())
    text = "\n\n".join([p for p in pages if p])
    # normalize spaces a bit
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def text_to_segments(text: str, approx_wpm: int = 160) -> List[Dict]:
    """
    Turn plain text into fake time-stamped 'segments' so your notes/quiz UI works.
    We estimate timing by reading speed (WPM).
    """
    if not text.strip():
        return []

    # split into sentences (very light)
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]

    secs_per_word = 60.0 / max(80, min(300, approx_wpm))  # clamp to sane range
    segments: List[Dict] = []
    cur_t = 0.0

    for s in sents:
        n_words = max(1, len(re.findall(r"[A-Za-z0-9]+", s)))
        dur = n_words * secs_per_word
        seg = {"start": float(cur_t), "end": float(cur_t + dur), "text": s}
        segments.append(seg)
        cur_t += dur

    return segments
