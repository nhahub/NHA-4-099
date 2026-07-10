# CareerGPS × AI Team — CV Evaluation API Contract
 
**Version:** 1.0.0  
**Last Updated:** 2026-07-01  
**Status:** Draft — pending AI Team confirmation
 
---
 
## Overview
 
CareerGPS sends a CV file (PDF) to the AI Team for ATS evaluation.  
The AI Team processes the file synchronously and returns the structured evaluation result in the response.
 
```text
CareerGPS Backend  →  POST /evaluate (multipart/form-data) →  AI Team
AI Team            →  evaluates CV (ATS compatibility, structure, content)
AI Team            →  HTTP 200 OK (JSON response)          →  CareerGPS Backend
```
 
---
 
## Authentication
 
All requests to the evaluation API must include a pre-shared API key in the header:
 
```http
X-API-Key: <shared-api-key>
```
 
- CareerGPS includes this header when calling the AI Team endpoint.
- If the header is missing or wrong → return `403 Forbidden` immediately, no processing.
 
---
 
## API Endpoint
 
### Evaluate CV
 
```http
POST https://ai-team-domain.com/evaluate
Content-Type: multipart/form-data
X-API-Key: <shared-api-key>
```
 
#### Request Body (form-data)
 
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `file` | ✅ | The CV PDF file to evaluate (Maximum size: 10MB) |
 
#### Expected Response — Success
 
```http
HTTP 200 OK
Content-Type: application/json
```
 
```json
{
  "scores": {
    "parseability_formatting": {
      "score": 28.5,
      "max": 30,
      "evidence": "Clean text extraction, standard fonts used."
    },
    "section_structure": {
      "score": 20.0,
      "max": 25,
      "evidence": "Missing clear 'Education' heading."
    },
    "content_quality": {
      "score": 22.0,
      "max": 25,
      "evidence": "Good action verbs, quantifiable achievements present."
    },
    "keyword_optimization": {
      "score": 15.0,
      "max": 20,
      "evidence": "Missing key industry skills."
    }
  },
  "deductions": {
    "total": 5.0,
    "reasons": "Spelling mistakes in the summary section."
  },
  "ats_report": {
    "has_multi_column_layout": false,
    "has_tables": false,
    "has_text_in_images": false,
    "is_scanned_pdf": false,
    "missing_sections": ["Education"],
    "contact_info_in_header_footer": false,
    "font_count": 2,
    "page_count": 1,
    "word_count": 450
  },
  "key_strengths": [
    "Strong use of quantifiable metrics",
    "Clean and readable formatting"
  ],
  "areas_for_improvement": [
    "Include a dedicated education section",
    "Add more industry-specific keywords"
  ],
  "total_score": 80.5,
  "total_max": 100
}
```
 
#### Error Responses
 
| Status | Meaning |
|---|---|
| `400` | Bad Request — Only PDF files are supported, or invalid format |
| `403` | Forbidden — Invalid or missing `X-API-Key` |
| `413` | Payload Too Large — File exceeds the 10MB limit |
| `429` | Rate Limited — Exceeded the limit of 5 requests per minute |
| `500` | Internal Server Error — Failed to evaluate the resume |
 
---
 
## Field Reference — Evaluation Data
 
### Root Object
 
| Field | Type | Required | Notes |
|---|---|---|---|
| `scores` | `Scores` | ✅ | Breakdown of scores by category |
| `deductions` | `Deductions` | ✅ | Any penalty points applied |
| `ats_report` | `ATSFormattingReport \| null` | ❌ | Technical ATS formatting details |
| `key_strengths` | `string[]` | ✅ | 1 to 5 items highlighting strengths |
| `areas_for_improvement` | `string[]` | ✅ | 1 to 3 items highlighting weaknesses |
| `total_score` | `float` | ✅ | Final calculated score (base score - deductions, min: 0, max: `total_max`) |
| `total_max` | `int` | ✅ | Maximum possible score (usually 100) |
 
### `Scores`
 
| Field | Type | Required | Max Value |
|---|---|---|---|
| `parseability_formatting` | `CategoryScore` | ✅ | 30 |
| `section_structure` | `CategoryScore` | ✅ | 25 |
| `content_quality` | `CategoryScore` | ✅ | 25 |
| `keyword_optimization` | `CategoryScore` | ✅ | 20 |
 
### `CategoryScore`
 
| Field | Type | Required | Notes |
|---|---|---|---|
| `score` | `float` | ✅ | Score achieved in this category |
| `max` | `int` | ✅ | Maximum possible score for this category |
| `evidence` | `string` | ✅ | Justification or evidence supporting the score |
 
### `Deductions`
 
| Field | Type | Required | Notes |
|---|---|---|---|
| `total` | `float` | ✅ | Total deduction points (max: 50.0) |
| `reasons` | `string` | ✅ | Detailed reasons for the applied deductions |
 
### `ATSFormattingReport`
 
| Field | Type | Required | Notes |
|---|---|---|---|
| `has_multi_column_layout` | `boolean` | ✅ | Whether the layout has multiple columns |
| `has_tables` | `boolean` | ✅ | Whether tables are present |
| `has_text_in_images` | `boolean` | ✅ | Whether text is embedded inside images |
| `is_scanned_pdf` | `boolean` | ✅ | Whether the PDF is a scanned image (no text layer) |
| `missing_sections` | `string[]` | ✅ | List of standard sections missing from the CV |
| `contact_info_in_header_footer` | `boolean` | ✅ | Whether contact info is placed in headers/footers |
| `font_count` | `int` | ✅ | Number of unique fonts used |
| `page_count` | `int` | ✅ | Number of pages |
| `word_count` | `int` | ✅ | Total word count |
 
---
 
## Constraints & Rules
 
| Rule | Detail |
|---|---|
| Max file size | 10MB |
| Supported formats | PDF only (bypassing this check results in a 400 or 500) |
| Rate Limit | 5 requests per minute per IP |
| Synchronous Timeout | CareerGPS should expect the response within the standard HTTP timeout window, as the evaluation is synchronous. |
 
---
 
## Environments
 
| Environment | AI Team Endpoint | Notes |
|---|---|---|
| Development | `http://localhost:8000/evaluate` | |
| Staging | TBD | |
| Production | TBD | |
 
---
 
## Questions for AI Team
 
- [ ] What are the endpoint URLs for staging and production?
- [ ] What is the expected average evaluation response time since this is a synchronous endpoint?
- [ ] Are we keeping the rate limit strictly to 5/minute for production?
