FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/voices
COPY . ./

RUN uv pip install --system -r requirements.txt

CMD ["python", "/app/oai.py"]