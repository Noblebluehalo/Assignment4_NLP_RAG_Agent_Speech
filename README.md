# CPSC 254 — Assignment #4: Text Classification, LLM RAG, Agents, and Speech

## Overview
This repository contains the six required Python source files for CPSC 254 Assignment #4:
- `topic_classifier.py` (Part 1: AG News loading, DTM/TF-IDF vectorization, sparsity analysis, and 4-configuration MLP experiments)
- `classification_agent.py` (Part 2: Zero-shot LangChain agent using `gpt-oss` on the Nautilus Hypercluster with JSON validation and a 200-sample stratified benchmark against the best Part 1 MLP)
- `rag.py` (Part 3.1: Syllabus PDF chunking, embedding, persistent Chroma vector store, and grounded RAG answering)
- `rag_eval.py` (Part 3.2: Evaluation script for syllabus-specific questions using the persisted Chroma index)
- `stt_rag.py` (Part 4.1: Speech-to-text transcription via Whisper paired with the syllabus RAG pipeline)
- `tts_rag.py` (Part 4.2: Syllabus RAG answers converted to spoken audio and played/saved)

## Environment Assumptions & Dependencies
- **OS & Python:** Tested on Windows with Python 3.14.7 (global Python environment; no virtual environment required).
- **Required Python Packages:**
  ```bash
  python -m pip install numpy scikit-learn joblib datasets torch
  python -m pip install python-dotenv langchain langchain-openai langchain-community langchain-text-splitters langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers
  python -m pip install openai-whisper sounddevice gTTS pyttsx3
  ```
  **Windows:** Whisper also requires the FFmpeg command-line tool to read audio files.
  Install it with `winget install --id Gyan.FFmpeg.Shared --exact`, then restart PowerShell.
  Verify it is available with `ffmpeg -version`.
  *(Note: `topic_classifier.py` automatically falls back to Hugging Face `datasets` when `torchtext.datasets.AG_NEWS` is unavailable or unsupported in the installed `torchtext` version.)*
- **Configuration (`.env` file in project folder — do NOT submit):**
  ```env
  OPENAI_API_KEY=your_nautilus_api_key
  NAUTILUS_BASE_URL=https://ellm.nrp-nautilus.io/v1
  NAUTILUS_MODEL=gpt-oss
  ```
- **Required Input File:** Place `CPSC 254_SYL.pdf` in the project root folder before running Parts 3 and 4.

## Run Instructions & Expected Commands

1. **Part 1 — Train & Evaluate MLP Topic Classifiers (`topic_classifier.py`)**
   ```bash
   python topic_classifier.py
   ```
   Generates dataset counts, example articles, training DTM shape and sparsity, metrics/confusion matrices across all four vectorizer and stop-word configurations, three misclassified test examples, and saves `part1_best_model.joblib`.

2. **Part 2 — Run Zero-Shot Classification Agent & 200-Sample Comparison (`classification_agent.py`)**
   ```bash
   python classification_agent.py
   ```
   Evaluates the `gpt-oss` LangChain agent and the saved Part 1 model on the same stratified sample of 200 test examples (`random_state=42`) and prints accuracy, latency, and validation results.

3. **Part 3 — Build/Query RAG Index & Run Evaluation (`rag.py`, `rag_eval.py`)**
   ```bash
   python rag_eval.py
   ```
   Builds `./chroma_syllabus_db` on first run (reusing it on later runs) and prints answers and retrieved syllabus chunks for evaluation. You can also run interactive Q&A via:
   ```bash
   python rag.py
   ```

4. **Part 4 — Speech-to-Text & Text-to-Speech RAG (`stt_rag.py`, `tts_rag.py`)**
   ```bash
   python tts_rag.py --question "What are the prerequisites for CPSC 254?"
   python stt_rag.py
   ```
   If microphone input is unavailable or silent, generate a test audio file and pass `--audio`.
   gTTS saves MP3 audio, so use an `.mp3` filename:
   ```bash
   python -c "from gtts import gTTS; gTTS('What is the late submission policy for CPSC 254?').save('sample_question.mp3')"
   python stt_rag.py --audio sample_question.mp3
   ```

## Submission Notes
- Submit the six `.py` files, this `README`, and your report (`.docx` or `.pdf`).
- Do **not** include `.env`, API keys, passwords, large datasets, or course sample scripts in the submission.