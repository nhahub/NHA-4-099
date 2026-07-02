"""
core/parsing_test.py — LLM-based CV/resume parser.

Calls a free Qwen instruct model through the Hugging Face Inference API
(hosted, remote — no weights are downloaded or run locally) to read raw
CV text and extract structured fields that conform to the ParsedData
schema defined in schemas.py.

Public API expected by main.py's lifespan/parse endpoint:
    load_models()   -> called once at FastAPI startup
    unload_models() -> called once at FastAPI shutdown
    parse(cv_text)  -> returns a ParsedData instance
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic import ValidationError

import config 
from core.Schemas import ParsedData

logger = logging.getLogger("parsing_test")

# ── Module-level client handle (populated by load_models) ──────────────────────
_client: Optional[InferenceClient] = None

_SCHEMA_JSON = json.dumps(ParsedData.model_json_schema(), indent=2)

_SYSTEM_PROMPT = f"""You are a precise resume/CV information extraction engine.

You will be given the raw text of a CV/resume. Extract the candidate's
information and respond with ONLY a single valid JSON object that strictly
conforms to the JSON Schema below.

Rules:
- Output raw JSON only — no markdown, no code fences, no commentary.
- If a field is not present in the CV, use null (or an empty list/object
  where the schema expects one).
- Never invent information that is not present in the text.
- Dates should be normalized to "YYYY-MM" or "YYYY" where possible, or
  "present" for ongoing roles.

JSON Schema:
{_SCHEMA_JSON}
"""


# ── Lifecycle ────────────────────────────────────────────────────────────────
def load_models() -> None:
    """Initializes the HF Inference API client. No weights are downloaded."""
    global _client

    if _client is not None:
        logger.info("Inference client already initialized, skipping.")
        return

    if not config.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a free token at "
            "https://huggingface.co/settings/tokens and export it as the "
            "HF_TOKEN environment variable."
        )

    logger.info("Initializing HF Inference API client for model=%s", config.MODEL_NAME)
    _client = InferenceClient(
        model=config.MODEL_NAME,
        token=config.HF_TOKEN,
        provider=config.PROVIDER,
        timeout=config.REQUEST_TIMEOUT,
    )
    logger.info("Inference client ready.")


def unload_models() -> None:
    """Releases the client. Safe to call even if never initialized."""
    global _client
    _client = None
    logger.info("Inference client released.")


# ── Prompt / generation helpers ─────────────────────────────────────────────
def _build_messages(cv_text: str) -> List[Dict[str, str]]:
    truncated = cv_text[: config.MAX_INPUT_CHARS]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'CV TEXT:\n"""\n{truncated}\n"""\n\nReturn only the JSON object.',
        },
    ]


def _generate(messages: List[Dict[str, str]]) -> str:
    if _client is None:
        raise RuntimeError("Inference client not initialized. Call load_models() first.")

    try:
        response = _client.chat_completion(
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )
    except HfHubHTTPError as exc:
        logger.error("HF Inference API call failed: %s", exc)
        raise RuntimeError(f"Hugging Face Inference API error: {exc}") from exc

    return response.choices[0].message.content.strip()


def _extract_json(raw_text: str) -> Dict[str, Any]:
    """Pulls the first {...} block out of the model's raw text output."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return json.loads(text[start : end + 1])


# ── Public API ───────────────────────────────────────────────────────────────
def parse(cv_text: str) -> ParsedData:
    """
    Calls the hosted Qwen model over `cv_text` and returns a validated
    ParsedData object. Retries with a corrective follow-up prompt if the
    model's output isn't valid JSON / doesn't match the schema.
    """
    if _client is None:
        raise RuntimeError("Inference client not initialized. Call load_models() first.")

    messages = _build_messages(cv_text)
    last_error: Optional[Exception] = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        raw_output = _generate(messages)
        try:
            data = _extract_json(raw_output)
            return ParsedData(**data)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Parse attempt %d/%d failed: %s", attempt, config.MAX_RETRIES, exc
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not valid JSON matching the schema. "
                        "Return ONLY the corrected JSON object, nothing else."
                    ),
                }
            )

    logger.error("All parse attempts failed for this CV: %s", last_error)
    raise RuntimeError(f"Failed to parse CV into structured data: {last_error}")