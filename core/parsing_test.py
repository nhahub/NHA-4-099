from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

import config
from core.Schemas import Certification, Education, Experience, Links, ParsedData, Project, Skills

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
    """No-op: this module uses the HF Inference API; there are no local weights to release."""
    logger.info("HF API client — nothing to unload.")


# ── HF API wrapper ─────────────────────────────────────────────────────────────

def _call_hf_api(model_id: str, task: str, json_data: dict, max_retries: int = 3) -> Any:
    """Call the HF Inference API with automatic retry on model-loading delays."""
    last_exc: Exception = RuntimeError("Unknown error")
    for i in range(max_retries):
        try:
            if task == "embedding":
                response = client.feature_extraction(json_data["inputs"], model=model_id)
                return response.tolist() if hasattr(response, "tolist") else response

            if task == "ner":
                return client.token_classification(
                    json_data["inputs"],
                    model=model_id,
                    aggregation_strategy=json_data.get("parameters", {}).get(
                        "aggregation_strategy", "simple"
                    ),
                )

            raise ValueError(f"Unknown task: {task!r}")

        except Exception as e:
            last_exc = e
            if i < max_retries - 1:
                err_msg = str(e).lower()
                delay = 5 if ("loading" in err_msg or "503" in err_msg) else 3
                time.sleep(delay)

    raise RuntimeError(f"HF API request failed after {max_retries} attempts: {last_exc}")


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _flatten_emb(emb: Any) -> List[float]:
    """Unwrap nested list wrappers until we reach a 1-D float list."""
    while isinstance(emb, list) and emb and isinstance(emb[0], list):
        emb = emb[0]
    return emb


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed texts in batches of 50. Returns one vector per input text."""
    results: List[List[float]] = []
    for i in range(0, len(texts), 50):
        chunk = texts[i : i + 50]
        raw = _call_hf_api(config.HF_EMBEDDING_URL, "embedding", {"inputs": chunk})
        if raw is None:
            results.extend([] for _ in chunk)
            continue
        # HF returns a flat vector for single-item batches, a list-of-vectors otherwise.
        if isinstance(raw, list) and raw and isinstance(raw[0], float):
            results.append(raw)
        else:
            results.extend(_flatten_emb(e) for e in raw)
    return results


# ── Cosine similarity ──────────────────────────────────────────────────────────

def _cos_sim(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


# ── Known languages (used by skills filter too) ───────────────────────────────

_KNOWN_LANGS = {
    "english", "arabic", "french", "german", "spanish", "italian",
    "portuguese", "russian", "chinese", "japanese", "korean", "turkish",
    "hindi", "dutch", "swedish", "polish", "czech", "greek", "hebrew",
    "persian", "urdu", "indonesian", "malay", "thai", "vietnamese",
}


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
_URL_RE = re.compile(
    r"https?://[^\s,;()]+"
    r"|(?:www\.)[^\s,;()]+\.[a-z]{2,}[^\s,;()]*"
    r"|\b[a-z0-9-]+\.(?:com|dev|io|me|net|org)/[^\s,;()]+",
    re.I,
)
_GPA_RE = re.compile(r"^\d(\.\d+)?\s*/\s*\d(\.\d+)?$")
_INSTITUTION_KEYWORDS = (
    "university", "college", "institute", "academy", "school of",
    "polytechnic",
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
        end = "Present" if end_raw in ("present", "current", "now") else _clean_date(m.group(2))
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
    "projects": [
        "projects", "personal projects", "key projects", "side projects",
        "academic projects", "project experience",
    ],
    "certifications": [
        "certifications", "certificates", "licenses certifications",
        "professional certifications", "courses certifications",
    ],
    "awards": [
        "awards", "awards activities", "honors", "honors awards",
        "achievements", "activities", "extracurricular activities",
    ],
}

_SIM_THRESHOLD = 0.60

# Headings are typically short ALL-CAPS or Title-Case phrases under 60 chars.
_HEADING_RE = re.compile(r"^[A-Z][A-Z\s/&]{2,58}$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    # Lines with digits or common punctuation are content, not headings.
    if re.search(r'[0-9@./\\:]', stripped):
        return False
    if stripped == stripped.upper() and re.search(r"[A-Z]", stripped):
        return True
    return bool(_HEADING_RE.match(stripped))


def _bucket_lines(lines: List[str]) -> Dict[str, List[str]]:
    """Assign each content line to the CV section it belongs to."""
    buckets: Dict[str, List[str]] = {k: [] for k in _SECTION_ANCHORS}
    if not lines:
        return buckets

    # Build anchor embeddings once up front.
    anchor_labels: List[str] = []
    anchor_texts: List[str] = []
    for label, anchors in _SECTION_ANCHORS.items():
        for a in anchors:
            anchor_labels.append(label)
            anchor_texts.append(a)

    try:
        anchor_embs = _get_embeddings(anchor_texts)
    except Exception as e:
        logger.error("Failed to embed section anchors: %s", e)
        return buckets

    # Only embed heading-like lines to avoid content lines stealing section labels.
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

    # Map each heading line to the best-matching section.
    heading_to_section: Dict[int, str] = {}
    for line_idx, emb in zip(heading_indices, heading_embs):
        if not emb:
            continue
        best_sim, best_label = -1.0, None
        for a_emb, a_label in zip(anchor_embs, anchor_labels):
            sim = _cos_sim(emb, a_emb)
            if sim > best_sim:
                best_sim, best_label = sim, a_label
        if best_sim >= _SIM_THRESHOLD and best_label:
            heading_to_section[line_idx] = best_label

    # Walk lines, track the active section, and assign content to its bucket.
    active_section: Optional[str] = None
    for i, line in enumerate(lines):
        if i in heading_to_section:
            active_section = heading_to_section[i]
            continue  # Don't include the raw heading in the bucket.
        if active_section:
            buckets[active_section].append(line)

    return buckets


# ── NER ────────────────────────────────────────────────────────────────────────

def _run_ner(text: str) -> Dict[str, List[str]]:
    """Run named-entity recognition over the full CV text."""
    out: Dict[str, List[str]] = {"PERSON": [], "ORG": [], "GPE": [], "LOC": []}
    if not text.strip():
        return out

    # Split into ~2 000-char chunks on line boundaries.
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

_TECH: set = {
    # Software / Web
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "r", "sql", "bash", "react",
    "vue", "angular", "svelte", "html", "css", "sass", "tailwind", "next.js",
    "nuxt", "webpack", "vite", "node.js", "django", "flask", "fastapi",
    "spring", "laravel", "express", "aws", "azure", "gcp", "docker",
    "kubernetes", "terraform", "git", "postgresql", "mysql", "mongodb",
    "redis", "elasticsearch", "machine learning", "deep learning",
    "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "spark", "kafka",
    # AI / ML / LLM
    "llm", "llms", "large language models", "generative ai", "gen ai",
    "prompt engineering", "fine-tuning", "rag", "langchain", "hugging face",
    "huggingface", "transformers", "nlp", "natural language processing",
    "computer vision", "object detection", "stable diffusion", "openai",
    "chatgpt", "gpt", "api integration", "rest api", "graphql",
    # Game Development
    "godot", "unity", "unreal engine", "pygame", "game development",
    "blender", "opengl", "vulkan",
    # Data / Analytics
    "power bi", "tableau", "excel", "looker", "dbt", "airflow", "snowflake",
    "bigquery", "databricks", "sas", "spss", "stata", "matlab",
    # Engineering / Industrial
    "autocad", "solidworks", "catia", "ansys", "simulink",
    "plc programming", "scada", "hvac", "cad", "cam", "cnc", "bim",
    "revit", "archicad", "etabs", "sap2000", "pvsyst",
    # Finance / Accounting
    "bloomberg", "sap", "oracle financials", "quickbooks", "xero",
    "financial modeling", "valuation", "ifrs", "gaap", "erp",
    # Healthcare / Medical
    "epic", "meditech", "cerner", "icd-10", "cpt coding", "ehr", "emr",
    "phlebotomy", "ecg", "radiology", "surgical techniques", "clinical trials",
    # Marketing / Design
    "google analytics", "seo", "sem", "hubspot", "salesforce", "mailchimp",
    "photoshop", "illustrator", "figma", "sketch", "indesign", "premiere pro",
    "after effects", "canva",
    # Legal / Compliance
    "legal research", "contract drafting", "westlaw", "lexisnexis",
    # Supply Chain / Logistics
    "wms", "sap mm", "supply chain management", "procurement",
    "inventory management", "lean", "six sigma", "kaizen",
    # Education
    "lms", "moodle", "blackboard", "google classroom",
}

_SOFT: set = {
    "teamwork", "communication", "leadership", "problem solving", "problem-solving",
    "critical thinking", "creativity", "adaptability", "time management",
    "negotiation", "collaboration", "organisation", "organization",
    "attention to detail", "analytical", "presentation", "mentoring",
    "coaching", "conflict resolution", "decision making",
    "classroom management", "data analysis", "technology integration",
    # Non-technical job soft skills
    "customer service", "patient care", "empathy", "active listening",
    "interpersonal skills", "multitasking", "stress management",
    "team building", "public speaking", "written communication",
    "verbal communication", "persuasion", "relationship management",
    "cultural awareness", "report writing", "record keeping", "case management",
    "community outreach", "stakeholder management", "facilitation",
}


def _build_wb_pattern(words: set) -> re.Pattern:
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])", re.I)

_TECH_RE = _build_wb_pattern(_TECH)
_SOFT_RE = _build_wb_pattern(_SOFT)

_LEVEL_KEYWORDS: Dict[str, str] = {
    "beginner": "Beginner", "basic": "Beginner", "elementary": "Beginner",
    "intermediate": "Intermediate", "proficient": "Intermediate",
    "advanced": "Advanced", "expert": "Advanced", "fluent": "Advanced",
    "native": "Advanced",
}
_LEVEL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _LEVEL_KEYWORDS) + r")\b",
    re.I,
)


def _strip_level(token: str) -> str:
    """Remove any inline proficiency qualifier (e.g. 'Fluent', 'Advanced') from a skill token."""
    return _LEVEL_RE.sub("", token).strip(" :-–()") or token.strip()


# Keywords that signal a token is an activity/role/description, not a skill.
_JUNK_ROLE_WORDS = {
    "ambassador", "instructor", "committee", "scout", "training",
    "achieved", "teaching", "taught", "participated", "volunteered",
    "organized", "represented", "attended", "completed", "certified",
    "offline", "online", "months", "students", "course", "program",
    "workshop", "session", "club", "chapter", "member", "association",
}


def _is_junk_skill_token(raw: str) -> bool:
    """True for tokens that aren't really skills (dates, GPAs, roles, descriptions)."""
    low = raw.strip().lower()
    if not low:
        return True

    # Date ranges and standalone years
    if _DATE_RANGE_RE.search(raw) or _GPA_RE.match(raw.strip()):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}(\s*[-–—]\s*(?:19|20)\d{2})?", raw.strip()):
        return True
    # Month + year (e.g. "Jan 2026")
    if re.fullmatch(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(?:19|20)\d{2}",
        low,
    ):
        return True
    # Year in parentheses at end, e.g. "AIESEC(2025)", "art committee (2024)"
    if re.search(r"\((?:19|20)\d{2}\)\s*$", raw.strip()):
        return True
    # Unclosed or malformed parentheses
    if raw.count("(") != raw.count(")"):
        return True
    # Fragment starting with conjunctions
    if re.match(r"^(?:and|or|&)\s+", low):
        return True
    # Institution names
    if any(kw in low for kw in _INSTITUTION_KEYWORDS):
        return True
    # Known spoken language
    if low in _KNOWN_LANGS:
        return True
    # Activity/role descriptors
    words = low.split()
    if any(w in _JUNK_ROLE_WORDS for w in words):
        return True
    # Too long to be a skill name (and not a known multi-word tech skill)
    word_count = len(words)
    if word_count > 5 or (raw.rstrip().endswith(".") and word_count > 3):
        return True
    return False


def _classify_skills(lines: List[str]) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    technical: List[Dict[str, Optional[str]]] = []
    non_technical: List[Dict[str, Optional[str]]] = []
    seen: set = set()

    for line in lines:
        cleaned = re.sub(r"^(?:[A-Za-z]{1,20}\s*){1,4}[:\-]\s*", "", line, count=1)
        # Split on commas, bullets, pipes, tabs, AND the word ' and ' between items.
        parts = re.split(r"[,|•·\t]+|\s+and\s+", cleaned)
        for part in parts:
            raw = part.strip(" -–•·\t()/")
            if not (2 <= len(raw) <= 80):
                continue
            if _is_junk_skill_token(raw):
                continue

            level_match = _LEVEL_RE.search(raw)
            level = _LEVEL_KEYWORDS[level_match.group(1).lower()] if level_match else None
            name = _strip_level(raw).strip()
            if not name or len(name) < 2 or _is_junk_skill_token(name):
                continue

            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            item = {"name": name, "level": level}

            if key in _SOFT or _SOFT_RE.search(key):
                non_technical.append(item)
            elif key in _TECH or _TECH_RE.search(key):
                technical.append(item)
            else:
                # Only treat as tool-like if it looks like a version string or
                # uses special chars (e.g. "C++", "node.js") — NOT plain
                # alphanumeric combos which catch event names like "Jan 2026".
                is_tool_like = bool(
                    re.search(r"[a-zA-Z]\.[a-zA-Z]|[+#@]", name)
                ) or bool(
                    re.search(r"v?\d+\.\d+", name)  # version numbers like "Python 3.11"
                )
                if is_tool_like:
                    technical.append(item)
                else:
                    non_technical.append(item)

    return technical, non_technical


# ── Experience ─────────────────────────────────────────────────────────────────

def _parse_experience(lines: List[str], orgs: List[str]) -> List[Experience]:
    blocks: List[Experience] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        start, end = _parse_date_range(line)
        matched_org = next(
            (o for o in orgs if len(o) > 2 and o.lower() in line.lower()), None
        )
        is_anchor = start is not None or matched_org is not None

        if is_anchor:
            if current:
                blocks.append(Experience(**current))

            raw_title = line
            if matched_org:
                raw_title = re.sub(re.escape(matched_org), "", raw_title, flags=re.I)
            if start:
                raw_title = _DATE_RANGE_RE.sub("", raw_title)
                raw_title = _DATE_RE.sub("", raw_title)
            title = raw_title.strip(" ,–-|") or None

            current = {
                "company": matched_org,
                "title": title,
                "startDate": start,
                "endDate": end,
                "description": None,
            }
        elif current is not None:
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

_MAJOR_RE = re.compile(r"^major\s*[:\-]\s*(.+)$", re.I)


def _parse_education(lines: List[str], orgs: List[str]) -> List[Education]:
    blocks: List[Education] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        start, end = _parse_date_range(line)
        dm = _DEGREE_RE.search(line)
        mm = _MAJOR_RE.match(line.strip())
        matched_org = next(
            (o for o in orgs if len(o) > 2 and o.lower() in line.lower()), None
        )
        is_anchor = dm is not None or matched_org is not None

        if mm and current is not None:
            current["major"] = mm.group(1).strip()
            continue

        if is_anchor:
            if current:
                blocks.append(Education(**current))
            current = {
                "institution": matched_org,
                "degree": dm.group(0).capitalize() if dm else None,
                "field": None,
                "major": None,
                "startDate": start,
                "endDate": end,
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


def _extract_languages(lines: List[str], fallback_text: str = "") -> List[str]:
    found: List[str] = []
    source = lines if lines else fallback_text.splitlines()
    for line in source:
        for word in re.split(r"[\s,/|•·\-]+", line):
            if word.strip().lower() in _KNOWN_LANGS:
                found.append(word.strip().capitalize())
    return list(dict.fromkeys(found))


# ── Projects ────────────────────────────────────────────────────────────────────

def _parse_projects(lines: List[str]) -> List[Project]:
    blocks: List[Project] = []
    current: Optional[Dict[str, Any]] = None

    def _finalize(d: Dict[str, Any]) -> Project:
        tech_found = {m.group(0) for m in _TECH_RE.finditer(d["tech_text"])}
        technologies = sorted(tech_found, key=str.lower)
        return Project(
            name=d["name"],
            description=(d["description"] or None),
            technologies=technologies,
            url=d["url"],
            startDate=d["startDate"],
            endDate=d["endDate"],
        )

    for line in lines:
        start, end = _parse_date_range(line)
        url_match = _URL_RE.search(line)
        is_new_block = start is not None or current is None

        if is_new_block:
            if current:
                blocks.append(_finalize(current))
            name = line
            if start:
                name = _DATE_RANGE_RE.sub("", name)
                name = _DATE_RE.sub("", name)
            if url_match:
                name = name.replace(url_match.group(0), "")
            name = re.sub(r"-\d{2}\b", "", name)
            name = name.strip(" ,–—-|:") or None
            current = {
                "name": name,
                "description": None,
                "tech_text": line,
                "url": url_match.group(0) if url_match else None,
                "startDate": start,
                "endDate": end,
            }
        else:
            current["description"] = ((current["description"] or "") + " " + line).strip()
            current["tech_text"] += " " + line
            if url_match and not current["url"]:
                current["url"] = url_match.group(0)
            if start and not current["startDate"]:
                current["startDate"] = start
            if end and not current["endDate"]:
                current["endDate"] = end

    if current:
        blocks.append(_finalize(current))
    return blocks


# ── Certifications ───────────────────────────────────────────────────────────────

def _parse_certifications(lines: List[str]) -> List[Certification]:
    certs: List[Certification] = []
    for line in lines:
        stripped = line.strip(" -–•·\t")
        if not stripped:
            continue
        date, _ = _parse_date_range(stripped)
        without_date = _DATE_RANGE_RE.sub("", stripped)
        without_date = _DATE_RE.sub("", without_date).strip(" ,–-|")

        parts = re.split(r"\s*[–\-|]\s*", without_date, maxsplit=1)
        name = parts[0].strip() or None
        issuer = parts[1].strip() if len(parts) > 1 else None
        certs.append(Certification(name=name, issuer=issuer or None, date=date))
    return certs


# ── Links ──────────────────────────────────────────────────────────────────────

def _extract_links(text: str) -> Optional[Links]:
    links: Dict[str, Optional[str]] = {"github": None, "linkedin": None, "portfolio": None}
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;)")
        if not url.lower().startswith(("http://", "https://")):
            url = "https://" + url
        low = url.lower()
        if "github.com" in low and not links["github"]:
            links["github"] = url
        elif "linkedin.com" in low and not links["linkedin"]:
            links["linkedin"] = url
        elif not links["portfolio"] and "github.com" not in low and "linkedin.com" not in low:
            links["portfolio"] = url
    return Links(**links) if any(links.values()) else None


# ── Public API ─────────────────────────────────────────────────────────────────

def parse(cv_text: str) -> ParsedData:
    """Parse raw CV text and return structured ParsedData."""
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]

    email = _extract_email(cv_text)
    phone = _extract_phone(cv_text)

    ner = _run_ner(cv_text)
    locations = ner["GPE"] + ner["LOC"]
    location_str = (
        f"{locations[0]}, {locations[1]}" if len(locations) >= 2
        else locations[0] if locations else None
    )

    buckets = _bucket_lines(lines)

    summary_lines = [l for l in buckets["summary"] if len(l) > 40] or buckets["summary"]
    summary = " ".join(summary_lines).strip() or None

    technical, non_technical = _classify_skills(buckets["skills"])
    experience = _parse_experience(buckets["experience"], ner["ORG"])
    projects = _parse_projects(buckets["projects"])
    education = _parse_education(buckets["education"], ner["ORG"])
    certifications = _parse_certifications(buckets["certifications"])
    languages = _extract_languages(buckets["languages"]) or _extract_languages([], cv_text)
    links = _extract_links(cv_text)

    return ParsedData(
        fullName=ner["PERSON"][0] if ner["PERSON"] else None,
        email=email,
        phone=phone,
        location=location_str,
        summary=summary,
        skills=Skills(technical=technical, nonTechnical=non_technical),
        experience=experience,
        projects=projects,
        education=education,
        certifications=certifications,
        languages=languages,
        links=links,
    )