# app/asr_engine.py
from __future__ import annotations

import os
import math
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

from pydub import AudioSegment
from faster_whisper import WhisperModel

# Optional: if present in requirements, we can fix LFS-pointer files automatically.
try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:
    hf_hub_download = None  # We'll guard usage below.

# ---- Constants / paths -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REQ_FILES = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]
# Heuristic: reject tiny Git LFS pointer files, require >1MB for model.bin
MIN_MODEL_BIN_BYTES = 1_000_000


# ---- Audio helpers -----------------------------------------------------------

def ensure_wav16k_mono(src_path: str, dst_path: str | None = None) -> str:
    """
    Converts ANY audio/video supported by ffmpeg (mp3/wav/m4a/mp4/webm, etc.)
    to 16kHz mono WAV for the ASR model.
    """
    dst_path = dst_path or (os.path.splitext(src_path)[0] + ".wav")
    AudioSegment.from_file(src_path).set_channels(1).set_frame_rate(16000).export(
        dst_path, format="wav"
    )
    return dst_path


def chunk_audio(wav_path: str, chunk_sec: int = 60) -> List[Tuple[int, float, float, str]]:
    """
    Slice a WAV into ~chunk_sec pieces, export each as its own WAV.
    Returns: list of (index, start_seconds, end_seconds, chunk_path)
    """
    audio = AudioSegment.from_wav(wav_path)
    chunks: List[Tuple[int, float, float, str]] = []
    total_ms = len(audio)
    n = max(1, math.ceil(total_ms / (chunk_sec * 1000)))
    for i in range(n):
        start_ms = i * chunk_sec * 1000
        end_ms = min((i + 1) * chunk_sec * 1000, total_ms)
        seg = audio[start_ms:end_ms]
        out = wav_path.replace(".wav", f".part{i:03}.wav")
        seg.export(out, format="wav")
        chunks.append((i, start_ms / 1000.0, end_ms / 1000.0, out))
    return chunks


# ---- Model loading (robust) --------------------------------------------------

def _local_model_dir(model_size: str) -> str:
    return os.path.join(MODELS_DIR, model_size)


def _is_real_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > MIN_MODEL_BIN_BYTES


def _has_real_local_model(model_dir: str) -> bool:
    return (
        os.path.isdir(model_dir)
        and all(os.path.isfile(os.path.join(model_dir, f)) for f in REQ_FILES)
        and _is_real_file(os.path.join(model_dir, "model.bin"))
    )


def _repair_local_model_if_needed(model_size: str) -> str:
    """
    If models/<size>/ exists but 'model.bin' is an LFS pointer/small file,
    try to download real artifacts from Hugging Face (optional).
    """
    model_dir = _local_model_dir(model_size)
    os.makedirs(model_dir, exist_ok=True)

    if _has_real_local_model(model_dir):
        return model_dir

    if hf_hub_download is None:
        # No repair possible without huggingface_hub
        return model_dir

    repo = f"Systran/faster-whisper-{model_size}"
    for fname in REQ_FILES:
        hf_hub_download(
            repo_id=repo,
            filename=fname,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
    return model_dir


@lru_cache(maxsize=1)
def get_model(model_size: str = "tiny") -> WhisperModel:
    """
    Load Faster-Whisper model with 3 paths:
      1) Use local models/<size>/ if REAL files exist (not LFS pointers);
      2) Attempt auto-repair via Hugging Face (if huggingface_hub installed);
      3) Fallback to model name (e.g., "tiny") to auto-download to cache.
    """
    local_dir = _local_model_dir(model_size)

    if _has_real_local_model(local_dir):
        source = local_dir
    else:
        repaired = _repair_local_model_if_needed(model_size)
        source = repaired if _has_real_local_model(repaired) else model_size

    # CPU-friendly defaults; remove/adjust for GPU
    # NOTE: do NOT set local_files_only=True, so fallback can download if needed.
    return WhisperModel(source, device="cpu", compute_type="int8")


# ---- Transcription -----------------------------------------------------------

def transcribe_fast(
    input_path: str,
    model_size: str = "tiny",
    chunk_sec: int = 60,
    on_progress: Callable[[int, int], None] | None = None,
) -> Tuple[str, List[Dict]]:
    """
    - Converts input to 16kHz mono WAV
    - Splits into chunks
    - Loads model (local if present & real; else auto-download)
    - Transcribes each chunk and stitches text with global timestamps
    Returns: (full_text, segments)
    """
    base_wav = ensure_wav16k_mono(input_path)
    parts = chunk_audio(base_wav, chunk_sec=chunk_sec)

    model = get_model(model_size)
    all_text: List[str] = []
    all_segs: List[Dict] = []
    total = len(parts)

    try:
        for idx, start_s, _end_s, path in parts:
            if on_progress:
                on_progress(idx, total)

            segments, _info = model.transcribe(path, beam_size=1)
            chunk_txt: List[str] = []
            for s in segments:
                txt = (s.text or "").strip()
                all_segs.append(
                    {
                        "start": float(start_s + (s.start or 0.0)),
                        "end": float(start_s + (s.end or 0.0)),
                        "text": txt,
                    }
                )
                if txt:
                    chunk_txt.append(txt)
            if chunk_txt:
                all_text.append(" ".join(chunk_txt))
    finally:
        # Cleanup chunk files and base wav
        for _i, _s, _e, p in parts:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.remove(base_wav)
        except Exception:
            pass

    return " ".join(all_text).strip(), all_segs
