# 🎤 Lecture Voice → Notes, Quiz & Concept Map

✨ An AI-powered interactive lecture assistant that converts lecture recordings, videos, or documents into structured notes, quizzes, and concept maps — helping students and professionals learn smarter and faster.

## 📖 Project Overview

Lectures and long documents often contain a wealth of information, but manually taking notes, making quizzes, and building study material can be time-consuming.
This project automates the process by combining speech-to-text transcription, natural language processing (NLP), and visualization tools.

Upload a lecture audio/video/PDF/TXT file and get:

Readable lecture notes

Interactive quizzes and flashcards

Concept maps and keyword visualizations


## 🎯 Objectives

✅ Build an end-to-end learning assistant from lecture input to study material output

✅ Improve student productivity by auto-generating structured notes

✅ Enable active learning via quizzes and flashcards

✅ Provide visual insights through concept maps & word clouds

✅ Support multiple input formats: audio, video, PDF, and text


## 🛠️ Tech Stack
~~~
Frontend/UI: Streamlit

Speech Recognition: Faster-Whisper
 (ONNX Runtime backend)

Document Processing: PyPDF2 / PDFMiner for PDFs, text parser for .txt

Notes & Quiz Generation: Custom NLP pipeline (keywords, summaries, question generation)

Visualization: Matplotlib, WordCloud for keyword clouds, Concept maps via co-occurrence graphs

Exports: Markdown, PDF (ReportLab), Anki CSV
~~~


## 🚀 Features

✅ Multi-format input – Upload mp3, wav, m4a, mp4, pdf, or txt files

✅ AI Transcription – Fast & accurate transcription with Faster-Whisper

✅ Auto Notes Generation – Evidence-backed, structured lecture notes

✅ Quiz Builder – Auto-generated MCQs, short-answer questions, and flashcards

✅ Concept Map – Visualize connections between key terms in your lecture

✅ Word Cloud – Quick visual summary of important keywords

✅ Export Options – Download notes as Markdown / PDF, and quizzes as Anki CSV




## 🔄 Project Workflow
~~~
flowchart TD
    A[🎤 Upload Lecture (Audio/Video/PDF/TXT)] --> B[🧠 Transcription / Text Extraction]
    B --> C[✍️ Notes Generator]
    B --> D[❓ Quiz & Flashcards Builder]
    B --> E[🗺️ Concept Map & Word Cloud]
    C --> F[⬇️ Export Notes (Markdown/PDF)]
    D --> G[⬇️ Export Quiz (Anki CSV)]
    E --> H[📊 Visualizations for Insights]
~~~


## 📂 Project Structure
~~~
Lecture-Voice-to-Notes-Generator-
│── app/
│   ├── asr_engine.py       # Speech recognition engine
│   ├── notes.py            # Notes generation logic
│   ├── quizzes.py          # Quiz & flashcard builder
│   ├── doc_ingest.py       # PDF/TXT ingestion
│── web/
│   ├── app.py              # Streamlit frontend
│── requirements.txt
│── README.md
~~~

## ⚙️ Installation

1.Clone the repository
~~~
git clone https://github.com/Sathwik-Surineni/Lecture-Voice-to-Notes-Generator-.git
cd Lecture-Voice-to-Notes-Generator-
~~~
2.Create and activate a virtual environment
~~~
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux

~~~
3.Install dependencies
~~~
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
~~~
4.Download Faster-Whisper model (Tiny)
~~~
Create a folder: models/tiny/

Place the following 4 files inside:

config.json (~2 KB)

tokenizer.json (~2–3 MB)

vocabulary.txt (~460 KB)

model.bin (~484 MB)
(Download from Systran/faster-whisper-tiny)
~~~
## Usage

Run the Streamlit app:
~~~
streamlit run web/app.py
~~~
Then open in your browser: http://localhost:8501



## Deployment Link 
~~~
https://sathwik-surineni-lecture-voice-to-notes-generator-webapp-twsgsx.streamlit.app/
~~~


## 🔮 Future Improvements

🌍 Support multi-language transcription

🧾 Add summarization for very long lectures

🤝 Enable collaborative note sharing (study groups)

📲 Mobile-friendly interface for on-the-go learning

☁️ Deploy on Streamlit Cloud / Hugging Face Spaces

🧠 Add AI-powered Q&A assistant over transcripts


## 🤝 Contributing

Pull requests are welcome! If you’d like to add features or fix bugs, please fork the repo and create a PR.

## 👨‍💻 Author

Sathwik Surineni
🔗 GitHub

## ⭐ Support

If you like this project, don’t forget to star ⭐ the repo — it helps a lot!





# ⚡ "Don’t just listen to lectures, learn from them."
