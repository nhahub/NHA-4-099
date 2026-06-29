import io
import logging
import multiprocessing
import os
from typing import Any, Dict
import docx

import uvicorn
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

import config
import core.parsing_test as parsing_test
from loader import download_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("main")


# ── In-Memory Storage ──────────────────────────────────────────────────────────
results_store: Dict[str, Any] = {}


# ── File helpers ───────────────────────────────────────────────────────────────
def _extract_text_from_docx(data: bytes) -> str:
    """Extracts raw text from .docx binary data."""
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_from_bytes(data: bytes, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _extract_text_from_pdf(data)
    elif ext == ".txt":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("cp1252")
    elif ext == ".docx":
        return _extract_text_from_docx(data)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, or DOCX.")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Loading models …")
    parsing_test.load_models()
    logger.info("Models ready.")
    yield
    logger.info("Unloading models …")
    parsing_test.unload_models()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CareerGPS — CV Parsing Service",
    version="2.0.0",
    description=(
        "CV parsing service. "
        "POST a file URL to /parse, "
        "then POST cvId to /results to get the parsed data."
    ),
    lifespan=lifespan,
)


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error: %s", exc)
    return JSONResponse(status_code=422, content={"status": "error", "detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal server error"})

# ── UI ─────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.abspath(os.path.join("UI", "app.html"))
    if not os.path.exists(path):
        raise HTTPException(404, detail="app.html not found")
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("cp1252")
    return HTMLResponse(content=content)


# ── Request Models ─────────────────────────────────────────────────────────────

class UrlParseRequest(BaseModel):
    url: str
    cvId: str


class ResultRequest(BaseModel):
    cvId: str


# ── Parse Endpoint ─────────────────────────────────────────────────────────────

@app.post("/parse", tags=["CV Pipeline"])
async def parse_cv(request: UrlParseRequest) -> JSONResponse:
    try:
        content = download_file(request.url)
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not download file: {exc}")

    url_filename = request.url.split("?")[0].rstrip("/").split("/")[-1] or "downloaded_file"
    try:
        cv_text = _extract_text_from_bytes(content, url_filename)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    if len(cv_text.strip()) < 10:
        raise HTTPException(400, detail="Could not extract enough text from the downloaded file.")

    try:
        parsed_data = parsing_test.parse(cv_text)
        parsed_dict = parsed_data.model_dump() if hasattr(parsed_data, "model_dump") else parsed_data
        result = {
            "cvId": request.cvId,
            "status": "completed",
            "parsedData": parsed_dict,
        }
        logger.info("Parsing complete — cvId=%s", request.cvId)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        logger.error("Parsing failed — cvId=%s: %s", request.cvId, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "cvId": request.cvId,
                "status": "failed",
                "error": str(exc),
            }
        )


# ── Results Endpoint ───────────────────────────────────────────────────────────

@app.post("/results", tags=["CV Pipeline"])
def get_result(request: ResultRequest) -> JSONResponse:
    data = results_store.get(request.cvId)

    if not data:
        raise HTTPException(404, detail="Result not ready yet or cvId invalid")

    if data.get("status") == "processing":
        raise HTTPException(202, detail="Still processing, try again shortly")

    # Delete from memory after returning
    results_store.pop(request.cvId, None)

    return JSONResponse(status_code=200, content=data)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)