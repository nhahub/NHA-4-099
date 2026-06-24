import io
import json
import logging
import multiprocessing
import os

import uvicorn
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

import config  
import parsing_test
from loader import download_file
from Schemas import ParseAccepted

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("main")


# ── File helpers ───────────────────────────────────────────────────────────────

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
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")


# ── Storage ────────────────────────────────────────────────────────────────────

def _save_json_locally(cv_id: str, data: dict) -> None:
    safe = "".join(c for c in cv_id if c.isalnum() or c in ("-", "_")).strip() or "unknown_cv"
    path = os.path.join(config.OUTPUT_DIR, f"{safe}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("Saved JSON — path=%s", path)
    except Exception as exc:
        logger.error("Failed to save JSON — cvId=%s: %s", cv_id, exc)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info("Output directory: %s", os.path.abspath(config.OUTPUT_DIR))
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
        "Async CV parsing service. "
        "POST a Cloudinary URL to /parse, "
        "then poll /results/{cvId} for the result."
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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


def serve_html(filename: str = "app.html") -> HTMLResponse:
    path = os.path.abspath(os.path.join("UI", filename))
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"{filename} not found")
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("cp1252")
    return HTMLResponse(content=content)


@app.get("/", response_class=HTMLResponse)
def root():
    return serve_html()


# ── URL-based Parse endpoint ───────────────────────────────────────────────────

class UrlParseRequest(BaseModel):
    url: str
    cvId: str


@app.post(
    "/parse",
    response_model=ParseAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Provide a Cloudinary (or any direct) URL to a CV file for parsing",
    tags=["CV Pipeline"],
)
async def parse_cv(
    request: UrlParseRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    try:
        file_path = download_file(request.url)
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not download file: {exc}")

    with open(file_path, "rb") as f:
        content = f.read()

    # Use the original URL to determine file type, not the temp download path
    url_filename = request.url.split("?")[0].rstrip("/").split("/")[-1] or file_path
    try:
        cv_text = _extract_text_from_bytes(content, url_filename)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    if len(cv_text.strip()) < 10:
        raise HTTPException(400, detail="Could not extract enough text from the downloaded file.")

    async def _parse_and_save():
        try:
            logger.info("Parsing started — cvId=%s", request.cvId)
            parsed_data = parsing_test.parse(cv_text)
            
            if hasattr(parsed_data, "model_dump"):
                parsed_dict = parsed_data.model_dump()
            elif hasattr(parsed_data, "dict"):
                parsed_dict = parsed_data.dict()
            else:
                parsed_dict = parsed_data
                
            _save_json_locally(request.cvId, {"cvId": request.cvId, "status": "completed", "parsedData": parsed_dict})
            logger.info("Parsing complete — cvId=%s", request.cvId)
        except Exception as exc:
            logger.error("Parsing failed — cvId=%s: %s", request.cvId, exc, exc_info=True)
            _save_json_locally(request.cvId, {"cvId": request.cvId, "status": "failed", "error": str(exc)})

    background_tasks.add_task(_parse_and_save)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=ParseAccepted(cvId=request.cvId).model_dump(),
    )


# ── Fetch Results endpoint ─────────────────────────────────────────────────────

class ResultRequest(BaseModel):
    cvId: str


@app.post("/results", summary="Retrieve and print parsing result by cvId", tags=["CV Pipeline"])
def get_result(request: ResultRequest):
    safe = "".join(c for c in request.cvId if c.isalnum() or c in ("-", "_")).strip()
    path = os.path.join(config.OUTPUT_DIR, f"{safe}.json")
    
    if not os.path.exists(path):
        raise HTTPException(404, detail="Result not ready yet or cvId invalid")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(500, detail="Result file is corrupted or incomplete.")
        
    # Pretty-print the data inside your server terminal console
    print("\n" + "="*50 + f"\n[PARSED DATA FOR cvId: {request.cvId}]\n" + "="*50)
    print(json.dumps(result_data, indent=4, ensure_ascii=False))
    print("="*50 + "\n")
    
    return result_data


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)