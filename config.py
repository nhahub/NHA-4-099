"""
config.py — global configuration for the CV parsing service.

Uses the Hugging Face Inference API (serverless, hosted) instead of loading
any model weights locally — nothing is downloaded or run on this machine.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────
# Any free-tier Qwen instruct model available on the HF Inference API.
# Swap as needed:
#   - "Qwen/Qwen2.5-7B-Instruct"   -> good accuracy/speed balance (default)
#   - "Qwen/Qwen2.5-72B-Instruct"  -> best quality, slower / may need PRO credits
#   - "Qwen/Qwen2.5-1.5B-Instruct" -> fastest, lowest accuracy
MODEL_NAME = "openai/gpt-oss-safeguard-20b"

# Required. Create a free token at https://huggingface.co/settings/tokens
# and set it as an environment variable — never hardcode it here.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Inference provider routed through huggingface_hub's InferenceClient.
# "auto" lets HF pick an available free provider for the model; you can
# pin one explicitly (e.g. "hf-inference", "together", "fireworks-ai") if
# you have a preference / quota on a specific provider.
PROVIDER = "auto"

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 1500
TEMPERATURE = 0.0        # deterministic extraction — no sampling randomness
TOP_P = 1.0
REQUEST_TIMEOUT = 60     # seconds per API call
MAX_RETRIES = 3          # re-prompt attempts if the model returns malformed JSON

# ── Input handling ────────────────────────────────────────────────────────────
# Truncate very long CVs so the prompt stays within the model's context window.
MAX_INPUT_CHARS = 12000