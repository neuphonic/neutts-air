FROM python:3.11-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    espeak-ng \
    git \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
COPY . .
# Models will be downloaded here
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

CMD ["python", "-m", "examples.basic_example", \
     "--input_text", "Hello, this is NeuTTS Air running inside Docker.", \
     "--ref_audio", "samples/dave.wav", \
     "--ref_text", "samples/dave.txt"]
