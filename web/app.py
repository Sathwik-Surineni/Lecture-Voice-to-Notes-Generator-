# web/app.py
import os, sys, io, tempfile, time
from typing import List, Dict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ReportLab (robust Unicode PDF)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

# ---------- Path + Imports ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.asr_engine import transcribe_fast
from app.notes import generate_notes, chunk_segments, build_notes_md
from app.quizzes import (
    generate_quiz_from_chunks as generate_quiz,
    quiz_to_anki_csv,
    quiz_ui,
)
from app.doc_ingest import extract_text_from_pdf, text_to_segments

# ---------- Streamlit page ----------
st.set_page_config(page_title="Interactive Lecture Assistant", page_icon="🎤", layout="wide")
st.title("🧑‍🏫 Lecture Voice → Notes, Quiz & Concept Map")

# ---------- Session State ----------
defaults = {
    "full_text": "",
    "segments": [],
    "notes_md": "",
    "quiz": {},
    "topic_window": 90,
    "bullets": 4,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------- Upload / Controls ----------
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader(
        "Upload audio/video/document (mp3/wav/m4a/mp4/pdf/txt)",
        type=["mp3", "wav", "m4a", "mp4", "pdf", "txt"],
        accept_multiple_files=False,
    )
with right:
    model_size = st.selectbox("ASR model size (local)", ["tiny", "small"], index=0)
    _vad = st.toggle("Skip silence (VAD est.)", value=False)  # UI only (current CPU path doesn’t use it)
chunk_sec = st.slider("ASR chunk size (sec)", 30, 120, 60, 1)

def _save_temp_file(up) -> str:
    suffix = "." + up.name.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(up.read())
    tmp.flush()
    tmp.close()
    return tmp.name

# ---------- Process Button ----------
if uploaded and st.button("🎬 Transcribe / Process", type="primary"):
    ext = uploaded.name.split(".")[-1].lower()

    # reset state
    st.session_state.full_text = ""
    st.session_state.segments = []
    st.session_state.notes_md = ""
    st.session_state.quiz = {}

    if ext in {"mp3", "wav", "m4a", "mp4"}:
        # AUDIO / VIDEO → ASR
        tmp_path = _save_temp_file(uploaded)
        prog = st.progress(0, text="Preparing…")

        def on_prog(i, total):
            pct = int(100 * (i + 1) / max(1, total))
            prog.progress(pct, text=f"Transcribing chunk {i+1}/{total}")

        with st.spinner("Transcribing audio/video…"):
            try:
                text, segments = transcribe_fast(
                    tmp_path,
                    model_size=model_size,
                    chunk_sec=chunk_sec,
                    on_progress=on_prog,  # safe if ignored by your function
                )
            finally:
                prog.empty()
                try:
                    os.remove(tmp_path)
                except:
                    pass

        st.session_state.full_text = text or ""
        st.session_state.segments = segments or []
        if text and text.strip():
            st.success("Transcription complete.")
        else:
            st.warning("No speech detected or empty result.")

    elif ext == "pdf":
        # PDF → text
        with st.spinner("Extracting text from PDF…"):
            pdf_bytes = uploaded.read()
            text = extract_text_from_pdf(pdf_bytes)
            segs = text_to_segments(text)  # synth segments to reuse pipeline (notes/quiz/map)
        st.session_state.full_text = text
        st.session_state.segments = segs
        st.success("PDF text extracted.")

    elif ext == "txt":
        # TXT → text
        with st.spinner("Reading text file…"):
            content = uploaded.read().decode("utf-8", errors="ignore")
            segs = text_to_segments(content)
        st.session_state.full_text = content
        st.session_state.segments = segs
        st.success("TXT loaded.")

# ---------- Transcript Preview ----------
if st.session_state.full_text:
    with st.expander("📰 Transcript / Document Text (read-only)", expanded=False):
        st.text_area("Preview", st.session_state.full_text, height=350, label_visibility="collapsed")

# ---------- Notes controls ----------
st.markdown("## 🧠 Evidence-backed Notes")
c1, c2 = st.columns(2)
with c1:
    st.session_state.topic_window = st.slider(
        "Topic window (sec) for notes/quiz/map",
        60, 180, st.session_state.topic_window, 5
    )
with c2:
    st.session_state.bullets = st.slider("Bullets per topic", 3, 8, st.session_state.bullets, 1)

# ---------- Generate Notes ----------
if st.session_state.segments and st.button("📝 Generate Notes"):
    with st.spinner("Building notes…"):
        chunks = chunk_segments(
            st.session_state.segments,
            max_window_sec=st.session_state.topic_window,
            overlap_sec=10,
        )
        st.session_state.notes_md = build_notes_md(
            chunks,
            bullets_per_chunk=st.session_state.bullets
        )
    st.success("Notes ready.")

# Show notes + WordCloud + Exports
if st.session_state.notes_md:
    st.markdown(st.session_state.notes_md)

    # ---- Word Cloud (from notes) ----
    try:
        txt = st.session_state.notes_md
        cloud = WordCloud(width=1200, height=450, background_color=None, mode="RGBA").generate(txt)
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.info(f"(Word cloud skipped: {e})")

    # ---- Exports: Markdown + PDF ----
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "⬇️ Download Notes (Markdown)",
            data=st.session_state.notes_md.encode("utf-8"),
            file_name="notes.md",
            mime="text/markdown",
        )
    with colB:
        # Build simple Unicode PDF using ReportLab
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=36,
            bottomMargin=36,
            leftMargin=36,
            rightMargin=36,
        )
        styles = getSampleStyleSheet()
        para = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_LEFT,
        )
        story = []
        for line in st.session_state.notes_md.splitlines():
            if line.startswith("# "):
                story.append(Paragraph(f"<b>{line[2:].strip()}</b>", styles["Heading1"]))
            elif line.startswith("## "):
                story.append(Paragraph(f"<b>{line[3:].strip()}</b>", styles["Heading2"]))
            elif line.startswith("- "):
                story.append(Paragraph("• " + line[2:].strip(), para))
            else:
                story.append(Paragraph(line, para))
            story.append(Spacer(1, 6))
        try:
            doc.build(story)
            pdf_bytes = buf.getvalue()
            st.download_button(
                "🧾 Download Notes (PDF)",
                data=pdf_bytes,
                file_name="notes.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF export skipped: {e}")

st.divider()

# ---------- Quiz ----------
st.markdown("## 🧩 Auto-Quiz & Flashcards (Interactive)")
if st.session_state.segments and st.button("🎯 Generate Quiz"):
    with st.spinner("Composing questions…"):
        chunks = chunk_segments(
            st.session_state.segments,
            max_window_sec=st.session_state.topic_window,
            overlap_sec=10,
        )
        st.session_state.quiz = generate_quiz(chunks)
    st.success("Quiz ready.")

if st.session_state.quiz:
    quiz_ui(st.session_state.quiz)
    csv_bytes = quiz_to_anki_csv(st.session_state.quiz)
    st.download_button(
        "⬇️ Export Quiz/Flashcards (Anki CSV)",
        data=csv_bytes,
        file_name="quiz_flashcards.csv",
        mime="text/csv",
    )

st.divider()

# ---------- Concept Map ----------
st.markdown("## 🗺️ Concept Map (topics ↔ key terms)")
if st.button("🧭 Generate Concept Map"):
    if not st.session_state.segments:
        st.warning("Transcribe/process a document first.")
    else:
        # Build co-occurrence matrix on top keywords from the transcript
        text = " ".join(s["text"] for s in st.session_state.segments)
        words = [w.lower() for w in __import__("re").findall(r"[A-Za-z][A-Za-z0-9\-]+", text)]
        stop = set("""
            a an the and or but if on in to for with by of from is are was were be been being that this those these
            as into about over under between among through during without within at it its it's they them he she you we
        """.split())
        words = [w for w in words if w not in stop and len(w) > 2]
        if not words:
            st.info("Not enough terms to make a concept map.")
        else:
            keys, counts = np.unique(words, return_counts=True)
            top_idx = np.argsort(-counts)[:20]
            vocab = keys[top_idx]

            # simple co-occurrence in sliding window
            co = np.zeros((len(vocab), len(vocab)), dtype=int)
            win = 12
            for i in range(len(words)):
                if words[i] not in vocab:
                    continue
                a = int(np.where(vocab == words[i])[0][0])
                for j in range(i + 1, min(i + win, len(words))):
                    if words[j] not in vocab:
                        continue
                    b = int(np.where(vocab == words[j])[0][0])
                    co[a, b] += 1
                    co[b, a] += 1

            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(co, interpolation="nearest")
            ax.set_xticks(range(len(vocab)))
            ax.set_xticklabels(vocab, rotation=90)
            ax.set_yticks(range(len(vocab)))
            ax.set_yticklabels(vocab)
            ax.set_title("Concept co-occurrence (higher = stronger link)")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
