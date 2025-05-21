FROM python:3.10-slim

RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

ARG GROQ_API_KEY

ENV GROQ_API_KEY=${GROQ_API_KEY}

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app/main.py"]