FROM python:3.11-slim-bookworm

WORKDIR /app

# Install git, espeak
RUN apt-get update && apt-get install -y git espeak-ng && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y build-essential python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install llama-cpp-python onnxruntime
RUN pip install uvicorn FastAPI

COPY models/ ./models/
# COPY samples/ ./samples/
COPY neuttsair/ ./neuttsair/
# COPY examples/basic_example.py ./basic_example.py

# just for now so can peek inside dockerfile
CMD ["uvicorn", "neuttsair.neutts:app", "--host", "0.0.0.0", "--port", "80"] 