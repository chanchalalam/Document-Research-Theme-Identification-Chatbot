# 1. Base Image
FROM python:3.10-slim

# Install Tesseract OCR and other system dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Argument for Groq API Key (to be passed during docker build)
ARG GROQ_API_KEY
# Set Groq API Key as an environment variable in the container
ENV GROQ_API_KEY=${GROQ_API_KEY}

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app/main.py"]