from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import time
import requests

import config
from Schemas import Education, Experience, ParsedData, Skills

logger = logging.getLogger("parser")

# ── HF Inference API ───────────────────────────────────────────────────────────
_sentence_model = None
_nlp = None


def _hf_post(url: str, payload: dict, retries: int = 3) -> Any:
    for attempt in range(retries):
        response = requests.post(url, headers=config.HF_HEADERS, json=payload, timeout=60)
        if response.status_code == 503:
            wait = int(response.headers.get("X-Wait-For-Model", 20))
            logger.warning("Model loading, waiting %ds (attempt %d)…", wait, attempt + 1)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"HF API unavailable after {retries} retries: {url}")


def load_models() -> None:
    logger.info("Using HF Inference API — no local models needed.")


def unload_models() -> None:
    logger.info("HF Inference API — nothing to unload.")


# ── Regex ──────────────────────────────────────────────────────────────────────
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
_SECTION_ANCHORS: Dict[str, List[str]] = {
    "summary":    ["professional summary", "about me", "career objective", "profile overview"],
    "skills":     ["technical skills", "programming languages", "core competencies", "tools technologies"],
    "experience": ["work experience", "employment history", "professional experience", "job responsibilities"],
    "education":  ["education background", "university degree", "academic history", "studied at"],
    "languages":  ["languages spoken", "language proficiency", "spoken written languages"],
}
_SIM_THRESHOLD = 0.35


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-9)


def _bucket_lines(lines: List[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {k: [] for k in _SECTION_ANCHORS}
    if not lines:
        return buckets

    anchor_labels, anchor_texts = [], []
    for label, anchors in _SECTION_ANCHORS.items():
        for a in anchors:
            anchor_labels.append(label)
            anchor_texts.append(a)

    try:
        all_embeddings = _hf_post(
            config.HF_EMBEDDING_URL,
            {"inputs": anchor_texts + lines, "options": {"wait_for_model": True}},
        )
        anchor_embs = all_embeddings[:len(anchor_texts)]
        line_embs   = all_embeddings[len(anchor_texts):]

        for i, line in enumerate(lines):
            best_idx   = max(range(len(anchor_embs)), key=lambda j: _cosine_sim(line_embs[i], anchor_embs[j]))
            best_score = _cosine_sim(line_embs[i], anchor_embs[best_idx])
            if best_score >= _SIM_THRESHOLD:
                buckets[anchor_labels[best_idx]].append(line)
    except Exception as exc:
        logger.error("Embedding API failed: %s", exc)

    return buckets


# ── NER ────────────────────────────────────────────────────────────────────────
def _run_ner(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"PERSON": [], "ORG": [], "GPE": [], "LOC": []}
    try:
        entities = _hf_post(
            config.HF_NER_URL,
            {"inputs": text[:512], "options": {"wait_for_model": True}},
        )
        label_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "GPE"}
        current_entity: Dict[str, str] = {}
        for ent in entities:
            label = ent.get("entity", "")
            word  = ent.get("word", "").replace("##", "")
            tag   = label.split("-")[-1]
            key   = label_map.get(tag)
            if not key:
                continue
            if label.startswith("B-") or not current_entity:
                if current_entity:
                    k = current_entity["key"]
                    v = current_entity["value"].strip()
                    if v and v not in out[k]:
                        out[k].append(v)
                current_entity = {"key": key, "value": word}
            else:
                current_entity["value"] += word
        if current_entity:
            k = current_entity["key"]
            v = current_entity["value"].strip()
            if v and v not in out[k]:
                out[k].append(v)
    except Exception as exc:
        logger.error("NER API failed: %s", exc)
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
    "coaching", "conflict resolution", "decision making",
}


def _classify_skills(lines: List[str]) -> Tuple[List[str], List[str]]:
    technical, non_technical = [], []
    for line in lines:
        # Strip only a leading "Label: " or "Label - " prefix (word chars up to first colon/dash)
        cleaned = re.sub(r"^[A-Za-z ]{1,30}[:\-]\s*", "", line, count=1)
        for part in re.split(r"[,|•·\t]+", cleaned):
            p = part.strip(" -–•·\t")
            if not (2 <= len(p) <= 60):
                continue
            if p.lower() in _SOFT or any(s in p.lower() for s in _SOFT):
                non_technical.append(p)
            else:
                technical.append(p)
    return technical, non_technical


# ── Experience ─────────────────────────────────────────────────────────────────
def _parse_experience(lines: List[str], orgs: List[str]) -> List[Experience]:
    blocks: List[Experience] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        start, end = _parse_date_range(line)
        is_anchor = start is not None or any(o.lower() in line.lower() for o in orgs if len(o) > 2)

        if is_anchor:
            if current:
                blocks.append(Experience(**current))
            current = {
                "company":     next((o for o in orgs if o.lower() in line.lower()), None),
                "title":       None,
                "startDate":   start,
                "endDate":     end,
                "description": None,
            }
        elif current is not None:
            if current["title"] is None and len(line) < 80:
                current["title"] = line
            else:
                current["description"] = ((current["description"] or "") + " " + line).strip()

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
        is_anchor = dm is not None or any(o.lower() in line.lower() for o in orgs if len(o) > 2)

        if is_anchor:
            if current:
                blocks.append(Education(**current))
            current = {
                "institution": next((o for o in orgs if o.lower() in line.lower()), None),
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

    email = _extract_email(cv_text)
    phone = _extract_phone(cv_text)

    ner = _run_ner(cv_text)
    locations    = ner["GPE"] + ner["LOC"]
    location_str = (
        f"{locations[0]}, {locations[1]}" if len(locations) >= 2
        else locations[0] if locations else None
    )

    buckets = _bucket_lines(lines)

    summary_lines = [l for l in buckets["summary"] if len(l) > 40] or buckets["summary"]
    summary = " ".join(summary_lines[:3]).strip() or None

    technical, non_technical = _classify_skills(buckets["skills"])
    experience = _parse_experience(buckets["experience"], ner["ORG"])
    education  = _parse_education(buckets["education"],  ner["ORG"])
    languages  = _extract_languages(buckets["languages"]) or _extract_languages([], cv_text)

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