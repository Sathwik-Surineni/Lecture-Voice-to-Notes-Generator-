from __future__ import annotations
from typing import List, Dict, Tuple
import csv, io, re, hashlib, random
from collections import Counter
import streamlit as st

# ------------------------
# Text helpers
# ------------------------

def _sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)

def _keywords(text: str, k: int = 20) -> List[str]:
    stop = set("""
        a an the and or but if on in to for with by of from is are was were be been being that this those these
        as into about over under between among through during without within at it its it's they them he she you we
    """.split())
    toks = [w.lower() for w in _words(text)]
    toks = [t for t in toks if t not in stop and len(t) > 2]
    freq = Counter(toks)
    return [w for w, _ in freq.most_common(k)]

def _fmt_ts(s: float, e: float) -> str:
    def f(x):
        x = int(x); h = x // 3600; m = (x % 3600) // 60; sec = x % 60
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
    return f"{f(s)}–{f(e)}"

# ------------------------
# Term picking
# ------------------------

def _pick_terms_from_chunks(chunks: List[Dict], top_k: int = 25) -> List[Tuple[str, str]]:
    big = " ".join(ch["text"] for ch in chunks)
    keys = _keywords(big, k=top_k * 2)
    seen, terms = set(), []
    for ch in chunks:
        first_sent = _sentences(ch["text"])[0] if ch["text"] else ""
        for w in _words(first_sent)[:6]:
            t = w.lower()
            if t not in seen and t in keys and len(t) > 2:
                seen.add(t); terms.append((t, first_sent))
                if len(terms) >= top_k:
                    return terms
    for t in keys:
        if t not in seen:
            seen.add(t); terms.append((t, ""))
        if len(terms) >= top_k:
            break
    return terms

# ------------------------
# Quiz generation
# ------------------------

def _stable_rng(seed_text: str) -> random.Random:
    """Create a per-question RNG that's stable within a run."""
    # Derive a deterministic 32-bit seed from text
    md5 = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    seed_int = int(md5[:8], 16)
    return random.Random(seed_int)

def generate_quiz_from_chunks(chunks: List[Dict]) -> Dict:
    mcqs, shorts, cards = [], [], []

    # Sentence pool with timestamps
    sent_pool: List[Tuple[str, float, float]] = []
    for ch in chunks:
        for s in _sentences(ch["text"]):
            sent_pool.append((s, ch["start"], ch["end"]))

    terms = _pick_terms_from_chunks(chunks, top_k=30)

    # ----- MCQs (randomized order) -----
    for term, _hint in terms:
        # find a sentence that contains the term
        found = next(((s, s0, s1) for (s, s0, s1) in sent_pool
                      if re.search(rf"\b{re.escape(term)}\b", s, re.I)), None)
        if not found:
            continue

        correct, s0, s1 = found

        # candidate distractors = sentences NOT containing the term and not identical to correct
        candidates = [s for (s, _, _) in sent_pool
                      if term.lower() not in s.lower() and s.strip() != correct.strip()]

        if len(candidates) < 3:
            continue

        # Use a stable RNG based on the term + timestamps so order doesn't jump around on re-renders
        rng = _stable_rng(f"{term}|{s0:.3f}|{s1:.3f}")

        # pick 3 random distractors
        distractors = rng.sample(candidates, 3)

        # build and shuffle choices
        choices = [correct] + distractors
        rng.shuffle(choices)

        mcqs.append({
            "question": f"What best describes **{term}**?",
            "choices": choices,             # randomized
            "answer": correct,              # still compare by string equality
            "ts": _fmt_ts(s0, s1),
        })
        if len(mcqs) >= 6:
            break

    # ----- Short answers -----
    for s, s0, s1 in sent_pool[:60]:
        prompt = re.sub(r"^(The|This|It)\s+", "", s).strip()
        if len(prompt.split()) < 6:
            continue
        shorts.append({
            "question": f"Explain: {prompt}",
            "answer": s,
            "ts": _fmt_ts(s0, s1),
        })
        if len(shorts) >= 3:
            break

    # ----- Flashcards -----
    for term, _ in terms[:20]:
        found = next(((s, s0, s1) for (s, s0, s1) in sent_pool
                      if re.search(rf"\b{re.escape(term)}\b", s, re.I)), None)
        if not found:
            continue
        s, s0, s1 = found
        cards.append({"front": term, "back": f"{s}  [{_fmt_ts(s0, s1)}]"})
        if len(cards) >= 5:
            break

    return {"mcq": mcqs, "short": shorts, "cards": cards}

# ------------------------
# Export
# ------------------------

def quiz_to_anki_csv(quiz: Dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Front", "Back", "Tags"])
    for q in quiz.get("mcq", []):
        choices = "\n".join([f"- {c}" for c in q["choices"]])
        w.writerow([f"{q['question']}  \n{choices}", f"{q['answer']}  \n(Source: {q['ts']})", "lecture;mcq"])
    for q in quiz.get("short", []):
        w.writerow([q["question"], f"{q['answer']}  \n(Source: {q['ts']})", "lecture;short"])
    for c in quiz.get("cards", []):
        w.writerow([c["front"], c["back"], "lecture;flashcard"])
    return buf.getvalue().encode("utf-8")

# ------------------------
# Streamlit UI
# ------------------------

def quiz_ui(quiz: Dict):
    st.write("### Multiple-choice")
    correct_count = 0
    for i, q in enumerate(quiz.get("mcq", []), 1):
        with st.container(border=True):
            st.markdown(
                f"**Q{i}.** {q['question']}  \n"
                f"<span style='opacity:0.6'>(source {q['ts']})</span>",
                unsafe_allow_html=True
            )
            key = f"mcq_{i}"
            choice = st.radio(
                " ",
                q["choices"],
                key=key,
                label_visibility="collapsed",
                index=None
            )
            if choice is not None:
                if choice == q["answer"]:
                    st.success("✅ Correct!")
                    correct_count += 1
                else:
                    st.error("❌ Incorrect.")
                with st.expander("Show answer"):
                    st.markdown(q["answer"])
    st.caption(f"MCQ score: **{correct_count}/{len(quiz.get('mcq', []))}**")

    st.write("---")
    st.write("### Short answer")
    for i, q in enumerate(quiz.get("short", []), 1):
        with st.container(border=True):
            st.markdown(
                f"**S{i}.** {q['question']}  \n"
                f"<span style='opacity:0.6'>(source {q['ts']})</span>",
                unsafe_allow_html=True
            )
            ans = st.text_input("Your answer", key=f"short_{i}")
            if ans:
                # very light match check
                ok = sum(w in q["answer"].lower() for w in set(_words(ans.lower()))) >= 2
                st.success("Looks reasonable ✅" if ok else "Not quite, check the reference ❗")
            with st.expander("Show reference"):
                st.markdown(q["answer"])

    st.write("---")
    st.write("### Flashcards")
    for i, c in enumerate(quiz.get("cards", []), 1):
        with st.expander(f"🃏 {c['front']}"):
            st.markdown(c["back"])
