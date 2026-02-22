FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_HOME=/app/weights/.torch \
    HF_HOME=/app/weights/.huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir -r requirements-cpu.txt
RUN pip install --no-cache-dir --no-deps git+https://github.com/piddnad/DDColor.git

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
