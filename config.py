import os
from dotenv import load_dotenv

load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Hugging Face Token ─────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it to your environment or .env file.")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ── HF Inference API URLs ──────────────────────────────────────────────────────
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_NER_URL       = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"