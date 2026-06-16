# NOVAC

NOVAC is an AI-powered document fraud detection and authenticity analysis system. It verifies uploaded documents using OCR, document quality checks, AI/synthetic document detection, forensic localization, visual tampering detection, text consistency checks, and automated PDF investigation reports.

The system is designed to analyze documents such as Aadhaar cards, PAN cards, ID documents, certificates, and scanned PDFs, then return a structured fraud risk assessment with visual evidence and extracted fields.

---

## Features

* Document upload and analysis
* OCR-based text extraction using PaddleOCR
* Structured field extraction for Aadhaar, PAN, and similar ID documents
* Document quality analysis for blur, glare, folds, creases, wrinkles, and readability
* AI-generated / synthetic document detection
* Masked-field and hidden-field detection
* ELA-based compression consistency analysis
* TruFor-based forgery localization
* MVSS-based visual tampering detection
* Text consistency and field mismatch detection
* Suspicious region visualization
* Real-time progress tracking during analysis
* MongoDB-backed result storage and history
* Professional PDF investigation report generation
* Frontend dashboard for upload, results, history, and report download

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

## Main Analysis Pipeline

NOVAC processes each uploaded document through the following stages:

```txt
Upload received
↓
File preparation
↓
PDF text extraction / PDF rendering
↓
OCR extraction
↓
Structured field matching
↓
Masked-field detection
↓
Document quality analysis
↓
Document authenticity / synthetic detection
↓
ELA analysis
↓
MVSS visual tampering analysis
↓
TruFor forgery localization
↓
Text consistency analysis
↓
Detector fusion
↓
Final risk calculation
↓
MongoDB result storage
↓
PDF report generation on request
```

---

## Detector Modules

### 1. OCR and Structured Field Extraction

The OCR module extracts text from images and scanned documents. The post-processing layer converts raw OCR lines into structured fields.

Supported structured fields include:

* Document type
* Name
* Date of birth
* Gender
* Aadhaar number
* VID
* PAN number
* Father's name
* Address
* Enrolment number
* Issue date

Raw OCR lines are cleaned and filtered before being displayed in the final report.

---

### 2. Document Quality Analysis

The quality module checks whether the uploaded document is clear enough for automated analysis.

It detects:

* Severe blur
* Glare or overexposure
* Low resolution
* Physical folds
* Creases
* Wrinkles
* Torn or damaged document regions
* Unclear document boundaries

Document quality is treated separately from fraud risk. A document can be high risk due to fraud, or simply unclear due to quality issues.

---

### 3. Document Authenticity and Synthetic Detection

This module checks whether the document appears authentic or AI-generated.

It detects:

* Synthetic / AI-generated documents
* Overly clean fake templates
* Placeholder-like ID numbers
* Weak camera or print acquisition traces
* Suspicious digital composition
* Official digital PDF structure when applicable

Official digital PDFs are handled separately so that clean PDF rendering is not incorrectly treated as AI-generated.

---

### 4. Masked Field Detection

The masking module detects hidden or masked critical fields, such as masked Aadhaar numbers or covered document identifiers.

Examples:

```txt
XXXX XXXX XXXX
**** **** ****
xxxxxxxxxxxx
```

Masked fields are highlighted in the annotated image when the image is otherwise analyzable.

---

### 5. ELA Analysis

ELA checks image compression consistency and helps identify suspicious edited regions based on abnormal compression artifacts.

---

### 6. TruFor Forgery Localization

TruFor is used for image forgery localization. It identifies possible manipulated regions in the uploaded document image.

The system includes:

* Persistent model loading
* Detector result caching
* Timeout handling
* Detailed timing logs
* Standalone runtime test support

---

### 7. MVSS Visual Tampering Detection

MVSS is used for visual tampering detection. It is kept in a separate environment because it may require specific PyTorch dependencies.

The system includes:

* Separate `mvss_venv`
* CPU-safe MVSS execution
* Persistent worker reuse
* Detector result caching
* Timeout handling
* Progress updates during long inference

---

## Final Result Logic

NOVAC separates the final result into independent signals:

* Fraud risk
* Document authenticity
* Document quality
* Detector evidence
* Suspicious regions

This prevents one signal from incorrectly overriding all others.

Example:

```txt
Risk Level: High Risk
Quality Badge: Unclear Document
```

This means the document has fraud indicators, while also having quality issues.

---

## Decisive Early Signals

If the system detects a decisive issue early, it avoids unnecessary deep forensic processing.

Deep detectors such as MVSS and TruFor can be skipped or cancelled when:

* Masked critical fields are detected
* Document quality is poor or unprocessable
* AI-generated / synthetic document is detected

Skipped detectors are marked clearly as skipped, not treated as clean passes.

Example skipped detector output:

```json
{
  "enabled": true,
  "completed": false,
  "skipped": true,
  "skip_reason": "synthetic_detected",
  "score": null,
  "suspicious_regions": [],
  "status": "skipped_due_to_decisive_signal"
}
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
  "stage": "Running OCR",
  "progress": 35,
  "message": "Extracting readable text from the document",
  "complete": false,
  "error": null
}
```

Common progress stages:

```txt
Upload received
Preparing file
Extracting PDF text
Rendering PDF page
Running OCR
Checking document quality
Checking document authenticity
Running ELA analysis
Running MVSS analysis
Running TruFor analysis
Running text consistency analysis
Combining detector results
Calculating final risk
Saving result
Analysis complete
```

---

## PDF Reports

NOVAC generates professional investigation reports containing:

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
* Suspicious regions
* Structured extracted fields
* Cleaned extracted text
* Final verdict
* Disclaimer

Raw OCR dumps are hidden by default to keep the report clean. Debug OCR details can be enabled separately if needed.

---

## Installation

NOVAC uses multiple Python environments because some forensic models require different dependency versions.

### 1. Main Backend Environment

The main backend environment runs FastAPI, OCR, MongoDB, PDF handling, document quality, authenticity analysis, ELA, report generation, and service orchestration.

```bat
cd /d D:\novac
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

If a requirements file is incomplete, install the common packages manually:

```bat
pip install fastapi uvicorn python-multipart pymongo python-dotenv opencv-python numpy pillow pymupdf reportlab paddleocr paddlepaddle
```

Run a compile check:

```bat
python -m compileall backend\app backend\scripts
```

Start the backend:

```bat
cd /d D:\novac
venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

Alternative backend start command:

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

MVSS runs in its own environment to avoid dependency conflicts.

```bat
cd /d D:\novac
python -m venv mvss_venv
mvss_venv\Scripts\activate
```

Install MVSS dependencies according to the MVSS-Net requirements:

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

MVSS currently runs CPU-safe by default for compatibility.

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

### 4. Environment Variables

Create a `.env` file in the expected backend/project location.

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

MongoDB is used to store analysis results, case history, progress status, and report metadata.

Make sure MongoDB is running before starting the backend.

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

Check backend imports:

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

NOVAC returns a structured result containing:

```json
{
  "case_id": "NOVAC-XXXXXXX",
  "fraud_score": 80,
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
  "detector_results": {},
  "suspicious_regions": [],
  "final_verdict": "Document authenticity concern detected."
}
```

---

## Project Status

NOVAC is complete as a working document fraud detection prototype.

Completed modules:

* Backend API
* React frontend
* OCR extraction
* Structured field matching
* Document quality analysis
* AI/synthetic document detection
* Masked-field detection
* ELA detector
* TruFor integration
* MVSS integration
* Detector fusion
* Progress tracking
* MongoDB result storage
* History page
* Results page
* PDF report generation
* Performance optimizations
* Multi-environment setup documentation

---

## Disclaimer

NOVAC provides automated document risk analysis using visual, textual, and forensic signals. The output is an investigation aid and should not be treated as legal proof of fraud. Manual verification is recommended before making acceptance, rejection, or escalation decisions.
