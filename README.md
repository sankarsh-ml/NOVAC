# NOVAC FAKE DOCUMENT AI 

NOVAC FAKE DOCUMENT AI is an AI-powered document fraud detection and authenticity analysis system. It analyzes uploaded documents using OCR, document quality checks, masked-field detection, AI/synthetic document detection, ELA, MVSS, TruFor, text consistency checks, and automated PDF investigation reports.

The system is designed for documents such as Aadhaar cards, PAN cards, ID cards, certificates, and scanned PDFs.

---

## Features

* Document upload and analysis
* OCR-based text extraction using PaddleOCR
* Structured field extraction for Aadhaar, PAN, and similar ID documents
* Masked-field and hidden-field detection
* Document quality analysis for blur, glare, folds, creases, wrinkles, and readability
* AI-generated / synthetic document detection
* ELA-based compression consistency analysis
* MVSS-based visual tampering detection
* TruFor-based forgery localization
* Text consistency and field mismatch detection
* Suspicious region visualization
* Backend-driven progress tracking
* MongoDB-backed result storage and history
* Professional PDF investigation report generation
* React frontend for upload, results, history, and report download

---

## Tech Stack

### Backend

* Python
* FastAPI
* MongoDB
* PaddleOCR
* OpenCV
* PyMuPDF
* PyTorch
* TruFor
* MVSS-Net
* ReportLab

### Frontend

* React
* JavaScript
* CSS
* REST API integration

---

## Project Structure

```txt
novac/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── services/
│   │   └── models/
│   ├── scripts/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── venv/
├── mvss_venv/
├── uploads/
├── README.md
└── .env
```

---

## How NOVAC Works

NOVAC separates document analysis into three major areas:

1. **Document quality**
   Checks whether the image is clear enough for analysis.

2. **Document authenticity**
   Checks whether the document appears real, synthetic, AI-generated, or digitally fabricated.

3. **Fraud / tampering evidence**
   Uses forensic detectors and text checks to identify suspicious regions or inconsistencies.

This separation prevents quality problems, authenticity issues, and fraud indicators from being incorrectly treated as the same type of failure.

---

## Parallel Analysis Pipeline

NOVAC uses a parallelized analysis flow to reduce total runtime while still keeping the main forensic detectors in the pipeline.

MVSS is one of the slowest stages, so it is started early in the background. While MVSS is running, the backend continues with other independent checks such as OCR, document quality, authenticity, ELA, and text consistency.

### Normal Pipeline Flow

```txt
Upload received
↓
Prepare file and shared preprocessing
↓
Start MVSS in background
↓
Run OCR while MVSS continues
↓
Run masked-field detection
↓
Run document quality analysis
↓
Run document authenticity / AI detection
↓
Run ELA analysis
↓
Run text consistency analysis
↓
Check decisive early signals
↓
If no decisive signal:
    wait for MVSS result
    run TruFor
    combine all detector results
    calculate final risk
    save result
    complete analysis
```

### Decisive-Signal Shortcut Flow

If NOVAC detects an obvious decisive issue early, it avoids unnecessary deep forensic processing.

Decisive conditions include:

* Masked or hidden critical fields
* Poor / bad / unprocessable document quality
* AI-generated / synthetic document detection

In these cases:

```txt
Start MVSS in background
↓
Run early checks
↓
Decisive issue detected
↓
Cancel or ignore MVSS result
↓
Skip TruFor
↓
Calculate final result using decisive evidence
↓
Save result
↓
Complete analysis
```

This means MVSS and TruFor are still part of the normal full analysis path, but they are not wasted on documents that already have a clear rejection/escalation reason.

---

## Detector Execution Logic

### MVSS

MVSS is started early in the background because it is one of the slowest stages.

MVSS can be:

* Completed normally
* Served from detector cache
* Cancelled due to a decisive early signal
* Marked as skipped/cancelled if no longer needed

A skipped detector is **not** treated as a clean pass.

Example skipped MVSS output:

```json
{
  "enabled": true,
  "completed": false,
  "skipped": true,
  "cancelled": true,
  "skip_reason": "synthetic_detected",
  "score": null,
  "suspicious_regions": [],
  "status": "cancelled_due_to_decisive_signal"
}
```

### TruFor

TruFor runs after MVSS only if no decisive early signal is found.

TruFor is skipped when:

* Synthetic document is detected
* Masked critical fields are detected
* Document quality is poor or unprocessable

Example skipped TruFor output:

```json
{
  "enabled": true,
  "completed": false,
  "skipped": true,
  "cancelled": false,
  "skip_reason": "poor_quality",
  "score": null,
  "suspicious_regions": [],
  "status": "skipped_due_to_decisive_signal"
}
```

---

## Main Analysis Stages

```txt
1. Upload received
2. File preparation
3. PDF text extraction / PDF rendering
4. Shared preprocessing
5. MVSS starts in background
6. OCR extraction
7. Structured field matching
8. Masked-field detection
9. Document quality analysis
10. Document authenticity / synthetic detection
11. ELA analysis
12. Text consistency analysis
13. Decisive-signal check
14. MVSS completion or cancellation
15. TruFor analysis, if required
16. Detector fusion
17. Final risk calculation
18. MongoDB result storage
19. PDF report generation on request
```

---

## Progress Tracking

NOVAC includes backend-driven progress tracking.

The frontend polls:

```txt
GET /analysis/status/{case_id}
```

Example response:

```json
{
  "case_id": "NOVAC-D03C6239",
  "stage": "Running OCR while MVSS continues",
  "progress": 35,
  "message": "Extracting readable text while MVSS runs in background.",
  "complete": false,
  "error": null
}
```

Common progress stages:

```txt
Upload received
Preparing file
Preparing MVSS input
Running MVSS in background
Running OCR while MVSS continues
Checking document quality
Checking document authenticity
Running ELA analysis
Running text consistency analysis
Checking decisive early signals
Cancelling MVSS analysis
Skipping TruFor analysis
Waiting for MVSS result
Running TruFor analysis
Combining detector results
Calculating final risk
Saving result
Analysis complete
```

---

## Detector Modules

### OCR and Structured Field Extraction

The OCR module extracts text from uploaded document images and PDFs. OCR output is cleaned and converted into structured fields.

Supported structured fields include:

* Document type
* Name
* Date of birth
* Gender
* Aadhaar number
* VID
* PAN number
* Father’s name
* Address
* Enrolment number
* Issue date

Raw OCR dumps are hidden from the final report by default.

---

### Masked-Field Detection

The masking module detects hidden or masked critical fields.

Examples:

```txt
XXXX XXXX XXXX
**** **** ****
xxxxxxxxxxxx
```

If masking is detected, NOVAC highlights the masked field region in the annotated image, provided the document itself is clear enough for annotation.

---

### Document Quality Analysis

The quality module checks whether the uploaded document is clear and reliable enough for automated analysis.

It detects:

* Severe blur
* Glare
* Overexposure
* Low resolution
* Physical folds
* Creases
* Wrinkles
* Torn or damaged regions
* Poor readability

Poor quality is handled separately from fraud risk.

Example:

```txt
Risk Level: High Risk
Quality Badge: Unclear Document
```

This means the document has fraud indicators and also has quality issues.

---

### Document Authenticity and Synthetic Detection

This module checks whether a document appears real or synthetic.

It detects:

* AI-generated documents
* Synthetic templates
* Placeholder ID patterns
* Weak camera or print acquisition traces
* Suspicious digital composition
* Official digital PDF structure

Official digital PDFs are treated separately so that clean digital rendering is not incorrectly treated as AI-generated.

---

### ELA Analysis

ELA checks compression consistency and helps identify suspicious editing artifacts.

---

### MVSS Visual Tampering Detection

MVSS detects visual tampering and manipulated regions.

NOVAC runs MVSS in a separate environment because MVSS may require dependency versions different from the main backend.

MVSS includes:

* Separate `mvss_venv`
* CPU-safe execution
* Persistent worker reuse
* Detector result caching
* Cancellation support
* Timeout protection
* Progress updates

---

### TruFor Forgery Localization

TruFor detects possible forged or manipulated image regions.

NOVAC includes:

* Persistent model loading
* Detector result caching
* Timeout handling
* Runtime debugging scripts
* Detailed timing logs

TruFor is only run when the document has not already been decisively flagged by early checks.

---

## Final Result Logic

NOVAC keeps separate fields for:

* Fraud score
* Risk level
* Document authenticity
* Document quality
* Quality badge
* Detector evidence
* Suspicious regions
* Final verdict

Important behavior:

```txt
AI-generated document:
    Fraud Score = 100
    Deep detectors skipped/cancelled
    No region annotation required

Poor-quality document:
    Fraud Score = 50
    Quality badge shown
    Deep detectors skipped/cancelled
    No region annotation required

Masked-field document:
    Masked field is annotated
    Deep detectors skipped/cancelled if masking is decisive

Normal document:
    MVSS completes
    TruFor runs
    Full detector fusion is performed
```

---

## PDF Reports

NOVAC generates professional PDF investigation reports.

Reports include:

* Case ID
* File name
* Analysis date and time
* Risk level
* Fraud score
* Authenticity score
* Document quality score
* Quality badge
* Key findings
* Detector summary
* Suspicious regions, when applicable
* Structured extracted fields
* Possible detected values
* Cleaned extracted text
* Final verdict
* Disclaimer

The report does **not** show raw OCR dumps by default.

Raw OCR details can be enabled only in debug mode.

---

## Installation

NOVAC uses multiple Python environments because some forensic models require different dependency versions.

---

### 1. Main Backend Environment

The main backend environment runs:

* FastAPI
* OCR
* MongoDB integration
* PDF handling
* document quality checks
* authenticity checks
* ELA
* report generation
* pipeline orchestration

```bat
cd /d D:\novac
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

If the requirements file is incomplete, install common backend packages manually:

```bat
pip install fastapi uvicorn python-multipart pymongo python-dotenv opencv-python numpy pillow pymupdf reportlab paddleocr paddlepaddle
```

Compile check:

```bat
python -m compileall backend\app backend\scripts
```

Start backend:

```bat
cd /d D:\novac
venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

Alternative:

```bat
cd /d D:\novac\backend
..\venv\Scripts\activate
uvicorn app.main:app --reload
```

Backend runs at:

```txt
http://127.0.0.1:8000
```

---

### 2. MVSS Environment

MVSS runs in its own environment.

```bat
cd /d D:\novac
python -m venv mvss_venv
mvss_venv\Scripts\activate
```

Install MVSS dependencies:

```bat
pip install -r backend\MVSS-Net\requirements.txt
```

Check MVSS runtime:

```bat
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Run MVSS runtime test if available:

```bat
python backend\scripts\test_mvss_cpu_runtime.py
```

---

### 3. Frontend Setup

```bat
cd /d D:\novac\frontend
npm install
npm run dev
```

The frontend communicates with the backend for:

* Uploading documents
* Starting analysis
* Polling progress
* Viewing results
* Viewing history
* Downloading PDF reports

---

## Environment Variables

Create a `.env` file in the expected backend or project location.

Example:

```env
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=novac

MVSS_DEVICE=cpu
MVSS_TIMEOUT_SECONDS=300
TRUFOR_TIMEOUT_SECONDS=180

FULL_FORENSIC_MODE=true
DEBUG_REPORT=false
PARALLEL_DETECTORS=false
```

---

## MongoDB

MongoDB is used for:

* Case results
* Analysis history
* Progress status
* Report metadata
* Detector outputs

Make sure MongoDB is running before starting backend analysis.

---

## Running the Full Project

### Terminal 1 — Backend

```bat
cd /d D:\novac
venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

### Terminal 2 — Frontend

```bat
cd /d D:\novac\frontend
npm run dev
```

---

## Runtime Health Checks

Check backend runtime:

```bat
venv\Scripts\activate
python backend\scripts\check_runtime_health.py
```

Check MVSS runtime:

```bat
mvss_venv\Scripts\activate
python backend\scripts\test_mvss_cpu_runtime.py
```

Check TruFor runtime:

```bat
venv\Scripts\activate
python backend\scripts\test_trufor_runtime.py path\to\test_image.jpg
```

Compile backend:

```bat
python -m compileall backend\app backend\scripts
```

---

## Git Ignore

Do not commit environments, uploads, caches, logs, or generated files.

Recommended `.gitignore` entries:

```gitignore
venv/
mvss_venv/
__pycache__/
.pytest_cache/
.mypy_cache/
uploads/
*.pyc
*.pyo
*.pyd
.env
*.log
*.pdf
*.tmp
```

Large model weights and checkpoints should not be committed unless the repository is explicitly configured for them.

---

## Example Output

```json
{
  "case_id": "NOVAC-XXXXXXX",
  "fraud_score": 100,
  "risk_level": "Synthetic Document Suspected",
  "result_status": "synthetic_suspected",
  "document_quality": {
    "quality_status": "good",
    "quality_score": 79
  },
  "document_authenticity": {
    "synthetic_detected": true,
    "synthetic_score": 100,
    "authenticity_score": 0
  },
  "deep_detectors_skipped": true,
  "deep_skip_reason": "synthetic_detected",
  "skipped_detectors": ["mvss", "trufor"],
  "final_verdict": "Document authenticity concern detected."
}
```

---

## Project Status

NOVAC FAKE DOCUMENT AI is complete as a working document fraud detection prototype.

Completed modules:

* FastAPI backend
* React frontend
* OCR extraction
* Structured field matching
* Masked-field detection
* Document quality analysis
* AI/synthetic document detection
* ELA detector
* MVSS integration
* TruFor integration
* Detector fusion
* Parallel MVSS pipeline scheduling
* Decisive-signal cancellation logic
* Progress tracking
* MongoDB result storage
* History page
* Results page
* PDF report generation
* Multi-environment setup documentation

---

## Disclaimer

NOVAC FAKE DOCUMENT AI provides automated document risk analysis using visual, textual, and forensic signals. The output is an investigation aid and should not be treated as legal proof of fraud. Manual verification is recommended before making acceptance, rejection, or escalation decisions.
