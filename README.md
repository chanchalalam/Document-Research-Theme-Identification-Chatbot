# 📄 Python Backend Application for Document Q\&A and Summarization

This project is a Python-powered backend system that allows users to upload and interact with documents—including PDFs and images—through question answering and summarization functionalities. The solution harnesses OCR, FAISS for vector search, and the Groq API for advanced language modeling.

---

## 🚀 Features

* **📄 Multi-Format Document Processing:** Supports PDFs, text files, and image documents.
* **🔍 OCR Integration:** Uses Tesseract OCR to extract text from images or scanned PDFs.
* **🧠 LLM (Groq) Integration:** Connects to the Groq API to enable generative capabilities like answering questions and summarization.
* **🗃️ Vector Search with FAISS:** Embeds documents using Sentence Transformers and stores them in FAISS for efficient similarity-based querying.
* **📡 API Interface (FastAPI):** Exposes endpoints to upload, query, and fetch summaries.
* **🗂 MongoDB for Metadata:** All associated metadata for each document and user interaction is stored in MongoDB.

---

## ⚙️ Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| Language        | Python 3.10                             |
| Web Framework   | FastAPI                                 |
| OCR Engine      | Tesseract + pytesseract                 |
| LLM             | Groq API                                |
| NLP / Embedding | SentenceTransformers, LangChain         |
| Vector DB       | FAISS                                   |
| Database        | MongoDB                                 |
| Deployment      | Docker, Hugging Face (for Streamlit UI) |

---

## 📁 Project Structure

```
.
├── backend
│   ├── app
│   │   ├── api/         # API route definitions
│   │   ├── core/        # App configuration and utilities
│   │   ├── services/    # Business logic (OCR, FAISS, Groq)
│   │   ├── models/      # Data & Pydantic models
│   │   ├── main.py      # FastAPI entry point
│   │   └── config.py    # App settings and secrets
│   ├── Dockerfile       # Docker setup
│   └── requirements.txt # All dependencies
├── frontend
│   ├── app.py           # Streamlit frontend app
│   └── utils/           # Helper utilities for UI
├── data/                # Folder to store documents
├── demo/                # Examples and demonstration files
├── tests/               # Test cases
├── README.md            # Project documentation
```

---

## 🐳 Running with Docker

### 1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Set your Groq API Key:

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Ensure `.env` is in your `.gitignore`.

### 3. Build Docker image:

```bash
docker build --build-arg GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d '=' -f2) -t document-qa-backend .
```

### 4. Run the Docker container:

```bash
docker run -d -p 5000:5000 --name docqa document-qa-backend
```

Visit the app at: [http://localhost:5000](http://localhost:5000)



---
title: My Python Backend # You can change this title
emoji: 🐍
colorFrom: green
colorTo: blue
sdk: docker
app_file: Dockerfile
app_port: 5000
pinned: false
---

# My Python Backend Application

This space runs my Python backend application using Docker.

Make sure your `main.py` (or the entry point specified in the Dockerfile's CMD)
starts a web server on `0.0.0.0` and port `5000`. 
