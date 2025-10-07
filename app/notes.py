from __future__ import annotations
from typing import List, Dict
from datetime import timedelta
from collections import Counter
import re

def _fmt(ts: float) -> str:
    return str(timedelta(seconds=int(ts)))

def chunk_segments(segments: List[Dict], max_window_sec: int = 90, overlap_sec: int = 10) -> List[Dict]:
    if not segments:
        return []
    out, cur, cur_start, cur_end = [], [], segments[0]["start"], segments[0]["end"]
    for s in segments:
        if (s["end"] - cur_start) <= max_window_sec:
            cur.append(s["text"]); cur_end = s["end"]
        else:
            out.append({"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()})
            cur_start = max(cur_end - overlap_sec, s["start"])
            cur = [s["text"]]; cur_end = s["end"]
    if cur:
        out.append({"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()})
    return out

def _sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def _keyword_scores(text: str) -> Counter:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)]
    stop = set("""
        a an the and or but if on in to for with by of from is are was were be been being that this those these
        as into about over under between among through during without within at it its it's they them he she you we
    """.split())
    words = [w for w in words if w not in stop and len(w) > 2]
    return Counter(words)

def summarize_block(text: str, max_bullets: int = 4) -> List[str]:
    sents = _sentences(text)
    if len(sents) <= max_bullets:
        return sents
    scores = _keyword_scores(text)
    def score(sent: str) -> int:
        toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", sent)]
        return sum(scores.get(w, 0) for w in toks)
    ranked = sorted(sents, key=score, reverse=True)
    top_set = set(ranked[:max_bullets])
    return [s for s in sents if s in top_set][:max_bullets]

def build_notes_md(chunks: List[Dict], bullets_per_chunk: int = 4) -> str:
    lines = ["# 📝 Evidence-backed Notes", ""]
    for i, ch in enumerate(chunks, 1):
        span = f"[{_fmt(ch['start'])}–{_fmt(ch['end'])}]"
        lines.append(f"## Topic {i} {span}")
        bullets = summarize_block(ch["text"], max_bullets=bullets_per_chunk)
        if not bullets:
            lines.append("_(no salient sentences detected)_\n")
            continue
        for b in bullets:
            lines.append(f"- {b} {span}")
        lines.append("")
    return "\n".join(lines)

# Wrapper used by app.py
def generate_notes(segments: List[Dict], topic_window_sec: int = 90, bullets_per_topic: int = 4) -> str:
    chunks = chunk_segments(segments, max_window_sec=topic_window_sec, overlap_sec=10)
    return build_notes_md(chunks, bullets_per_chunk=bullets_per_topic)
