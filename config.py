import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Hugging Face Token ─────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it to your Space Secrets or .env file.")

# ── HF Inference Client ────────────────────────────────────────────────────────
client = InferenceClient(token=HF_TOKEN)

# ── Clean Model Strings (Updated) ──────────────────────────────────────────────
# Strip the 'https://api-inference.huggingface.co/models/' prefix completely
HF_EMBEDDING_URL = "sentence-transformers/all-MiniLM-L6-v2"
HF_NER_URL       = "Jean-Baptiste/roberta-large-ner-english"