📄 Document Research & Theme Identification Chatbot

This project is an interactive chatbot that can perform research across a large set of documents (minimum 75 documents), identify common themes (multiple themes are possible), and provide detailed, cited responses to user queries.

✅ Features

Multi-Format Document Processing: Supports PDFs, text files, and image documents.
OCR Integration: Uses Tesseract OCR to extract text from images or scanned PDFs.
LLM (Groq) Integration: Connects to the Groq API to enable generative capabilities like answering questions.
Vector Search with FAISS: Embeds documents using Sentence Transformers and stores them in FAISS for efficient similarity-based querying.
API Interface (FastAPI): Exposes endpoints to upload, query, and fetch summaries.
⚙️ Tech Stack

Layer	Technology
Language	Python 3.10
Web Framework	FastAPI
OCR Engine	Tesseract + pytesseract
LLM	Groq API
NLP / Embedding	SentenceTransformers, LangChain
Vector DB	FAISS
Deployment	Docker, Hugging Face
📁 Project Structure

├── backend
│   ├── app
│   │   ├── api/        
│   │   ├── core/        
│   │   ├── services/    
│   │   ├── main.py
│   │   ├── templates/
│   │   └── .env
│   ├── Dockerfile       
│   └── requirements.txt 
├── data/                
├── README.md           
🐳 Running with Docker

1. Clone the repository
git clone <repository_url>
cd <repository_name>
2. Set your Groq API Key
Create a .env file in the root:

GROQ_API_KEY=your_groq_api_key_here
Ensure .env is in your .gitignore.

3. Build Docker image
docker build --build-arg GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d '=' -f2) -t document-qa-backend .
4. Run the Docker container
docker run -d -p 5000:5000 --name docqa document-qa-backend
Visit the app at: http://localhost:5000 