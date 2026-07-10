---
title: CV ATSfeat:

colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# CV ATS Evaluator API

A robust FastAPI backend service designed to evaluate resume PDFs against deterministic Applicant Tracking System (ATS) formatting constraints and structural best practices. 

Instead of traditional keyword matching against a specific job description, this API acts as an ATS "linter"—flagging formatting choices that break ATS parsers (e.g., multi-column layouts, tables, text-in-images, missing core sections) and using LLMs to evaluate the qualitative structure and impact of the resume content.

## Features

- **Deterministic Parsing Checks**: Uses `PyMuPDF` to physically inspect the PDF structure and flag:
  - Multi-column layouts
  - Tables and complex bounding boxes
  - Text embedded in images
  - Scanned (no-text-layer) PDFs
  - Contact information hidden in document headers/footers
- **LLM Content Evaluation**: Uses Groq (or Google Gemini/Ollama) to evaluate the resume across 4 distinct categories:
  1. Parseability & Formatting
  2. Section Structure
  3. Content Quality
  4. Keyword Optimization
- **FastAPI Backend**: Fully asynchronous, scalable API built for modern Python environments.
- **Docker Ready**: Includes a `Dockerfile` for easy containerization and deployment to platforms like Hugging Face Spaces.

## Directory Structure

- `app/`: Core application logic, routers, LLM services, and Jinja prompts.
- `tests/`: API testing scripts.
- `data/`: CSV data exports and batch evaluation outputs.
- `synsetic_data/`: A suite of synthetic test PDFs used to benchmark the ATS formatting parser.

## Installation

1. Clone the repository.
2. Ensure you have Python 3.12+ installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and add your API key:
   ```bash
   cp .env.example .env
   ```
   *Note: Open `.env` and set `API_KEY` to your Groq API key.*

## Running Locally

Start the FastAPI server using Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You can now access the interactive Swagger API documentation at:
`http://localhost:8000/docs`

### Testing the API
You can use the provided test script to evaluate a synthetic resume:
```bash
python tests/test_api.py
```
