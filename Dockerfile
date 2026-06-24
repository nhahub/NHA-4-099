# Use a clean Python environment
FROM python:3.10-slim

# Prevent Python from buffering logs or writing bytecode
# Change this line in your Dockerfile:
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/code/hf_cache        # ← /code is writable during build and persists
WORKDIR /code

# Copy and install dependencies
COPY req.txt /code/req.txt
RUN pip install --no-cache-dir --upgrade -r /code/req.txt

# ── PRE-DOWNLOAD SPACH MODEL ──
RUN python -m spacy download en_core_web_sm

# ── PRE-DOWNLOAD LOCAL TRANSFORMER ──
# This script runs during the container build, saving the model in /tmp/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your app files
COPY . /code

# Fix file permissions for Hugging Face's non-root environment
RUN mkdir -p /code/hf_cache /tmp/parsed_output /tmp/data && chmod -R 777 /tmp /code/hf_cache
EXPOSE 7860

# Run your FastAPI main exactly on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]