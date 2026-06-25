from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import math
import time
from huggingface_hub import InferenceClient

import config
from Schemas import Education, Experience, ParsedData, Skills

logger = logging.getLogger("parser")

client = InferenceClient(token=config.HF_TOKEN)


# ── Model lifecycle ────────────────────────────────────────────────────────────

def load_models() -> None:
    logger.info("Pinging Hugging Face Inference APIs to wake them up...")
    try:
        client.feature_extraction("wake up", model=config.HF_EMBEDDING_URL)
        client.token_classification("wake up", model=config.HF_NER_URL)
        logger.info("HF APIs pinged.")
    except Exception as e:
        logger.warning("Failed to ping HF APIs: %s", e)


def unload_models() -> None:
    pass


# ── HF API wrapper ─────────────────────────────────────────────────────────────

def _call_hf_api(model_id: str, task: str, json_data: dict, max_retries: int = 3) -> Any:
    for i in range(max_retries):
        try:
            if task == "embedding":
                response = client.feature_extraction(json_data["inputs"], model=model_id)
                if hasattr(response, "tolist"):
                    return response.tolist()
                return response

            elif task == "ner":
                response = client.token_classification(
                    json_data["inputs"],
                    model=model_id,
                    aggregation_strategy=json_data.get("parameters", {}).get(
                        "aggregation_strategy", "simple"
                    ),
                )
                return response

        except Exception as e:
            err_msg = str(e).lower()
            if ("loading" in err_msg or "503" in err_msg) and i < max_retries - 1:
                time.sleep(5)
                continue
            if i < max_retries - 1:
                time.sleep(3)
                continue
            raise RuntimeError(f"Request failed: {e}")

    raise RuntimeError("Max retries exceeded for HF API")


# ── FIX 1: Safe embedding flattener ───────────────────────────────────────────
# HF returns [[0.1, 0.2, ...]] for single-input batches instead of [0.1, 0.2, ...].
# Unwrap until we reach a list of floats.

def _flatten_emb(emb: Any) -> List[float]:
    """Unwrap nested list wrappers until we reach a 1-D float list."""
    while isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
        emb = emb[0]
    return emb


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    
    results: List[List[float]] = []
    for i in range(0, len(texts), 50):
        chunk = texts[i : i + 50]
        raw = _call_hf_api(config.HF_EMBEDDING_URL, "embedding", {"inputs": chunk})
        # raw may be a single vector (float list) or a batch (list of float lists)
        if raw is None:
            results.extend([[] for _ in chunk])
            continue
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], float):
            # Single vector returned for a single-item chunk
            results.append(raw)
        else:
            results.extend([_flatten_emb(e) for e in raw])
    return results


# ── Cosine similarity ──────────────────────────────────────────────────────────

def _cos_sim(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# ── Regex helpers ──────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", re.I)
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)", re.I)
_DATE_RANGE_RE = re.compile(
    r"((?:\w+\.?\s*)?\d{4})\s*[-–—to]+\s*((?:\w+\.?\s*)?\d{4}|present|current|now)",
    re.I,
)
_DATE_RE = re.compile(
    r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(\d{4})\b",
    re.I,
)
_DEGREE_RE = re.compile(
    r"\b(bachelor|master|phd|ph\.d|mba|bsc|msc|b\.sc|m\.sc|associate|"
    r"diploma|certificate|doctor|doctoral)\b",
    re.I,
)
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _clean_date(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"([a-z]+)\.?\s*(\d{4})", raw, re.I)
    if m:
        return f"{m.group(2)}-{_MONTH_MAP.get(m.group(1)[:3].lower(), '01')}"
    m2 = re.match(r"(\d{4})", raw)
    return m2.group(1) if m2 else raw


def _parse_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _DATE_RANGE_RE.search(text)
    if m:
        end_raw = m.group(2).lower().strip()
        end = "present" if end_raw in ("present", "current", "now") else _clean_date(m.group(2))
        return _clean_date(m.group(1)), end
    m2 = _DATE_RE.search(text)
    return (_clean_date(m2.group(0)), None) if m2 else (None, None)


def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    if m:
        raw = m.group(1).strip()
        if len(re.sub(r"\D", "", raw)) >= 7:
            return raw
    return None


# ── Section bucketing ──────────────────────────────────────────────────────────

# FIX 6: Added "additional information" and more real-world heading variants
_SECTION_ANCHORS: Dict[str, List[str]] = {
    "summary": [
        "professional summary", "about me", "career objective",
        "profile overview", "summary", "professional profile", "overview",
    ],
    "skills": [
        "technical skills", "programming languages", "core competencies",
        "tools technologies", "skills", "additional information",
        "other information", "key skills", "areas of expertise",
    ],
    "experience": [
        "work experience", "employment history", "professional experience",
        "job responsibilities", "experience", "career history",
        "work history", "professional background",
    ],
    "education": [
        "education background", "university degree", "academic history",
        "studied at", "education", "academic background", "qualifications",
    ],
    "languages": [
        "languages spoken", "language proficiency", "spoken written languages",
        "languages", "additional information", "linguistic skills",
    ],
}

# FIX 5: Raised from 0.35 → 0.60 to prevent content lines matching anchors
_SIM_THRESHOLD = 0.60

# Headings are typically short, ALL-CAPS or Title Case, and under 60 chars.
_HEADING_RE = re.compile(r"^[A-Z][A-Z\s/&]{2,58}$")


def _is_heading(line: str) -> bool:
    
    stripped = line.strip()
    if len(stripped) > 60 or len(stripped) < 3:
        return False
    # All-caps line (e.g. "WORK EXPERIENCE") or Title Case short phrase
    if stripped == stripped.upper() and re.search(r"[A-Z]", stripped):
        return True
    if _HEADING_RE.match(stripped):
        return True
    return False


def _bucket_lines(lines: List[str]) -> Dict[str, List[str]]:
    
    buckets: Dict[str, List[str]] = {k: [] for k in _SECTION_ANCHORS}
    if not lines:
        return buckets

    # Build anchor embeddings once
    anchor_labels: List[str] = []
    anchor_texts: List[str] = []
    for label, anchors in _SECTION_ANCHORS.items():
        for a in anchors:
            anchor_labels.append(label)
            anchor_texts.append(a)

    try:
        anchor_embs = _get_embeddings(anchor_texts)
    except Exception as e:
        logger.error("Failed to embed anchors: %s", e)
        return buckets

    # FIX 2: Identify heading lines only
    heading_indices = [i for i, l in enumerate(lines) if _is_heading(l)]

    if not heading_indices:
        logger.warning("No heading lines detected — falling back to embedding all lines.")
        heading_indices = list(range(len(lines)))

    heading_texts = [lines[i] for i in heading_indices]

    try:
        heading_embs = _get_embeddings(heading_texts)
    except Exception as e:
        logger.error("Failed to embed heading lines: %s", e)
        return buckets

    # Map each heading line → best-matching section
    heading_to_section: Dict[int, str] = {}
    for idx, (line_idx, emb) in enumerate(zip(heading_indices, heading_embs)):
        if not emb:
            continue
        best_sim, best_label = -1.0, None
        for a_emb, a_label in zip(anchor_embs, anchor_labels):
            sim = _cos_sim(emb, a_emb)
            if sim > best_sim:
                best_sim = sim
                best_label = a_label
        if best_sim >= _SIM_THRESHOLD and best_label:
            heading_to_section[line_idx] = best_label

    # Pass 2: walk lines, track active section, assign content to bucket
    active_section: Optional[str] = None
    for i, line in enumerate(lines):
        if i in heading_to_section:
            active_section = heading_to_section[i]
            # Don't add the raw heading itself to the bucket
            continue
        if active_section:
            buckets[active_section].append(line)

    return buckets


# ── NER ────────────────────────────────────────────────────────────────────────

def _run_ner(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"PERSON": [], "ORG": [], "GPE": [], "LOC": []}
    if not text.strip():
        return out

    # Split into ~2000-char chunks on line boundaries
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0
    for line in text.splitlines():
        if current_len + len(line) > 2000:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # FIX 3: Process ALL chunks, not just the first 3
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            entities = _call_hf_api(
                config.HF_NER_URL,
                "ner",
                {"inputs": chunk, "parameters": {"aggregation_strategy": "simple"}},
            )
            for ent in entities:
                label = ent.get("entity_group", ent.get("entity", ""))
                word = ent.get("word", "").strip().lstrip("# ")
                if label in ("PER", "B-PER", "I-PER", "PERSON"):
                    out["PERSON"].append(word)
                elif label in ("ORG", "B-ORG", "I-ORG"):
                    out["ORG"].append(word)
                elif label in ("LOC", "B-LOC", "I-LOC", "GPE"):
                    out["GPE"].append(word)
        except Exception as e:
            logger.error("NER failed for chunk: %s", e)

    return out


# ── Skills ─────────────────────────────────────────────────────────────────────

_TECH = {
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "r", "sql", "bash", "react",
    "vue", "angular", "svelte", "html", "css", "sass", "tailwind", "next.js",
    "nuxt", "webpack", "vite", "node.js", "django", "flask", "fastapi",
    "spring", "laravel", "express", "aws", "azure", "gcp", "docker",
    "kubernetes", "terraform", "git", "postgresql", "mysql", "mongodb",
    "redis", "elasticsearch", "machine learning", "deep learning",
    "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "spark", "kafka",
}
_SOFT = {
    "teamwork", "communication", "leadership", "problem solving", "problem-solving",
    "critical thinking", "creativity", "adaptability", "time management",
    "negotiation", "collaboration", "organisation", "organization",
    "attention to detail", "analytical", "presentation", "mentoring",
    "coaching", "conflict resolution", "decision making", "curriculum development",
    "classroom management", "data analysis", "technology integration",
}


def _classify_skills(lines: List[str]) -> Tuple[List[str], List[str]]:
    technical, non_technical = [], []
    for line in lines:
        cleaned = re.sub(r"^[A-Za-z ]{1,30}[:\-]\s*", "", line, count=1)
        for part in re.split(r"[,|•·\t]+", cleaned):
            p = part.strip(" -–•·\t")
            if not (2 <= len(p) <= 60):
                continue
            if p.lower() in _SOFT or any(s in p.lower() for s in _SOFT):
                non_technical.append(p)
            elif p.lower() in _TECH or any(t in p.lower() for t in _TECH):
                technical.append(p)
            else:
                # Default unrecognised items to non-technical
                non_technical.append(p)
    return technical, non_technical


# ── Experience ─────────────────────────────────────────────────────────────────

def _parse_experience(lines: List[str], orgs: List[str]) -> List[Experience]:
    blocks: List[Experience] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        start, end = _parse_date_range(line)
        matched_org = next((o for o in orgs if len(o) > 2 and o.lower() in line.lower()), None)
        is_anchor = start is not None or matched_org is not None

        if is_anchor:
            if current:
                blocks.append(Experience(**current))

            # FIX 4: Extract title directly from the anchor line by stripping
            # the org name and date range — what remains is usually the job title.
            raw_title = line
            if matched_org:
                raw_title = re.sub(re.escape(matched_org), "", raw_title, flags=re.I)
            if start:
                raw_title = _DATE_RANGE_RE.sub("", raw_title)
                raw_title = _DATE_RE.sub("", raw_title)
            title = raw_title.strip(" ,–-|") or None

            current = {
                "company":     matched_org,
                "title":       title,
                "startDate":   start,
                "endDate":     end,
                "description": None,
            }
        elif current is not None:
            # If title still missing and line is short and has no date, use it
            if current["title"] is None and len(line) < 80 and not _DATE_RE.search(line):
                current["title"] = line
            else:
                current["description"] = (
                    ((current["description"] or "") + " " + line).strip()
                )

    if current:
        blocks.append(Experience(**current))
    return blocks or [Experience()]


# ── Education ──────────────────────────────────────────────────────────────────

def _parse_education(lines: List[str], orgs: List[str]) -> List[Education]:
    blocks: List[Education] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        start, end = _parse_date_range(line)
        dm = _DEGREE_RE.search(line)
        matched_org = next((o for o in orgs if len(o) > 2 and o.lower() in line.lower()), None)
        is_anchor = dm is not None or matched_org is not None

        if is_anchor:
            if current:
                blocks.append(Education(**current))
            current = {
                "institution": matched_org,
                "degree":      dm.group(0).capitalize() if dm else None,
                "field":       None,
                "startDate":   start,
                "endDate":     end,
            }
        elif current is not None:
            if current["field"] is None and len(line) < 80 and not re.search(r"\d{4}", line):
                current["field"] = line
            if start and not current["startDate"]:
                current["startDate"] = start
            if end and not current["endDate"]:
                current["endDate"] = end

    if current:
        blocks.append(Education(**current))
    return blocks or [Education()]


# ── Languages ──────────────────────────────────────────────────────────────────

_KNOWN_LANGS = {
    "english", "arabic", "french", "german", "spanish", "italian",
    "portuguese", "russian", "chinese", "japanese", "korean", "turkish",
    "hindi", "dutch", "swedish", "polish", "czech", "greek", "hebrew",
    "persian", "urdu", "indonesian", "malay", "thai", "vietnamese",
}


def _extract_languages(lines: Optional[List[str]], fallback_text: str = "") -> List[str]:
    found: List[str] = []
    source = lines if lines else fallback_text.splitlines()
    for line in source:
        for word in re.split(r"[\s,/|•·\-]+", line):
            if word.strip().lower() in _KNOWN_LANGS:
                found.append(word.strip().capitalize())
    return list(dict.fromkeys(found))


# ── Public API ─────────────────────────────────────────────────────────────────

def parse(cv_text: str) -> ParsedData:
    """Parse raw CV text → structured ParsedData."""
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]

    email    = _extract_email(cv_text)
    phone    = _extract_phone(cv_text)

    # FIX 3: NER now runs over the full text (no chunk[:3] cap)
    ner = _run_ner(cv_text)
    locations    = ner["GPE"] + ner["LOC"]
    location_str = (
        f"{locations[0]}, {locations[1]}" if len(locations) >= 2
        else locations[0] if locations else None
    )

    # FIX 2 + 5: Only heading lines are embedded; threshold raised to 0.60
    buckets = _bucket_lines(lines)

    summary_lines = [l for l in buckets["summary"] if len(l) > 40] or buckets["summary"]
    summary = " ".join(summary_lines[:3]).strip() or None

    technical, non_technical = _classify_skills(buckets["skills"])

    # FIX 4: Experience parser now extracts title from the anchor line
    experience = _parse_experience(buckets["experience"], ner["ORG"])
    education  = _parse_education(buckets["education"],  ner["ORG"])

    # so _extract_languages will find them even if the heading wasn't canonical.
    languages = _extract_languages(buckets["languages"]) or _extract_languages([], cv_text)

    return ParsedData(
        fullName=ner["PERSON"][0] if ner["PERSON"] else None,
        email=email,
        phone=phone,
        location=location_str,
        summary=summary,
        skills=Skills(technical=technical, nonTechnical=non_technical),
        experience=experience,
        education=education,
        languages=languages,
    )