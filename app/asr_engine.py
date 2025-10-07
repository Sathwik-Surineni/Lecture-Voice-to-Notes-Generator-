# app/asr_engine.py
import os, math
from functools import lru_cache
from typing import Callable, Dict, List, Tuple
from pydub import AudioSegment
from faster_whisper import WhisperModel

def ensure_wav16k_mono(src_path: str, dst_path: str | None = None) -> str:
    """
    Converts ANY audio/video supported by ffmpeg (mp3/wav/m4a/mp4, etc.)
    to 16kHz mono WAV for the ASR model.
    """
    dst_path = dst_path or (os.path.splitext(src_path)[0] + ".wav")
    AudioSegment.from_file(src_path).set_channels(1).set_frame_rate(16000).export(dst_path, format="wav")
    return dst_path

def chunk_audio(wav_path: str, chunk_sec: int = 60) -> List[Tuple[int, float, float, str]]:
    audio = AudioSegment.from_wav(wav_path)
    chunks = []
    total_ms = len(audio)
    n = max(1, math.ceil(total_ms / (chunk_sec * 1000)))
    for i in range(n):
        start_ms = i * chunk_sec * 1000
        end_ms = min((i+1) * chunk_sec * 1000, total_ms)
        seg = audio[start_ms:end_ms]
        out = wav_path.replace(".wav", f".part{i:03}.wav")
        seg.export(out, format="wav")
        chunks.append((i, start_ms/1000.0, end_ms/1000.0, out))
    return chunks

@lru_cache(maxsize=1)
def get_model(model_size: str = "tiny") -> WhisperModel:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", model_size)
    required = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]
    if not os.path.isdir(model_path) or any(not os.path.isfile(os.path.join(model_path, f)) for f in required):
        raise FileNotFoundError(f"Missing model files in {model_path}. Put {', '.join(required)} there.")
    # CPU-safe defaults (avoid CUDA/cuDNN drama)
    return WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)

def transcribe_fast(
    input_path: str,
    model_size: str = "tiny",
    chunk_sec: int = 60,
    on_progress: Callable[[int, int], None] | None = None
) -> Tuple[str, List[Dict]]:
    base_wav = ensure_wav16k_mono(input_path)
    parts = chunk_audio(base_wav, chunk_sec=chunk_sec)

    model = get_model(model_size)
    all_text, all_segs = [], []
    total = len(parts)
    for idx, start_s, _end_s, path in parts:
        if on_progress: on_progress(idx, total)
        segments, _info = model.transcribe(path, beam_size=1)
        chunk_txt = []
        for s in segments:
            all_segs.append({
                "start": float(start_s + s.start),
                "end": float(start_s + s.end),
                "text": s.text.strip()
            })
            chunk_txt.append(s.text.strip())
        all_text.append(" ".join(chunk_txt))
        try: os.remove(path)
        except: pass

    return " ".join([t for t in all_text if t]).strip(), all_segs
