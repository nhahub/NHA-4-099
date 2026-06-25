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
# The client handles the base URL and resolves network paths natively inside Spaces
client = InferenceClient(token=HF_TOKEN)

# Define your model names as strings instead of full URLs
NER_MODEL = "Jean-Baptiste/roberta-large-ner-english"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"