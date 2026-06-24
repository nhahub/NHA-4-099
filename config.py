import os
import cloudinary
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Force Hugging Face transformers cache to a writeable directory
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
