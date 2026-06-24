# Use a clean Python environment
FROM python:3.11-slim

# Prevent Python from buffering logs or writing bytecode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/code/hf_cache

WORKDIR /code

# Copy and install dependencies
COPY req.txt /code/req.txt
RUN pip install --no-cache-dir --upgrade -r /code/req.txt

# ── PRE-DOWNLOAD SPACY MODEL ──
RUN python -m spacy download en_core_web_sm

# ── PRE-DOWNLOAD SENTENCE TRANSFORMER ──
# Runs during build so the model is cached in /code/hf_cache (persists in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your app files
COPY . /code

# Fix file permissions for Hugging Face's non-root environment
RUN mkdir -p /code/hf_cache /tmp/parsed_output /tmp/data && chmod -R 777 /tmp /code/hf_cache

EXPOSE 7860

# Run your FastAPI app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]