# NOVAC

NOVAC is an AI-powered document authenticity and fraud detection platform. It analyzes uploaded documents using OCR, forensic localization, visual tampering detection, synthetic document detection, document quality checks, and automated risk reporting.

## Features

* OCR-based text extraction and structured field matching
* Document quality analysis for blur, glare, folds, creases, and readability
* Document authenticity and synthetic document detection
* TruFor-based forgery localization
* MVSS-based visual tampering analysis
* ELA/compression consistency analysis
* Text consistency and suspicious field detection
* Suspicious region visualization
* Real-time analysis progress tracking
* MongoDB-backed result storage and history
* Downloadable PDF investigation reports

## Tech Stack

### Backend

* Python
* FastAPI
* MongoDB
* PaddleOCR
* OpenCV
* PyTorch
* TruFor
* MVSS-Net
* ReportLab / PDF generation utilities

### Frontend

* React
* JavaScript
* CSS
* REST API integration

## Analysis Pipeline

The system processes each uploaded document through multiple stages:

1. File upload and preparation
2. PDF text extraction or PDF-to-image rendering
3. OCR text extraction
4. Structured field extraction
5. Document quality analysis
6. Document authenticity analysis
7. AI/synthetic document detection
8. ELA analysis
9. TruFor forgery localization
10. MVSS tampering detection
11. Text consistency analysis
12. Detector fusion and final risk calculation
13. Result storage
14. PDF report generation

## Risk Output

NOVAC separates different types of findings instead of treating them as the same issue:

* **Fraud Risk**: final fraud/tampering risk score
* **Document Authenticity**: synthetic or AI-generated document suspicion
* **Document Quality**: clarity, folds, damage, blur, or readability warnings
* **Detector Signals**: individual detector contributions
* **Suspicious Regions**: annotated visual evidence areas

This allows a document to be visually clear but still suspicious, or physically damaged while still analyzable.

## Progress Tracking

The backend provides real-time analysis status updates so the frontend can show the current running stage.

Example stages:

* Upload received
* Running OCR
* Checking document quality
* Checking document authenticity
* Running TruFor analysis
* Running MVSS analysis
* Combining detector results
* Saving result
* Analysis complete

## PDF Reports

NOVAC generates investigation reports containing:

* Case summary
* Risk level
* Fraud score
* Authenticity score
* Document quality score
* Detector summary
* Suspicious regions
* Extracted structured fields
* Cleaned OCR text
* Final verdict and disclaimer

## Installation and Environment Setup

NOVAC uses multiple Python virtual environments because some detector models require different dependency versions. The main backend runs the FastAPI server, while heavier forensic detectors such as MVSS can run through their own isolated environments.

This prevents dependency conflicts between OCR, PyTorch models, PaddleOCR, TruFor, MVSS, and the main API.

## Environment Structure

Recommended project structure:

```txt
novac/
├── backend/
│   ├── app/
│   ├── scripts/
│   └── requirements.txt
├── frontend/
├── venv/              # Main backend environment
├── mvss_venv/         # MVSS-specific environment
├── uploads/
├── README.md
└── .env
```

## 1. Main Backend Environment

The main backend environment runs:

* FastAPI server
* MongoDB integration
* OCR service
* PDF processing
* document quality checks
* document authenticity checks
* ELA
* TruFor service interface
* MVSS service interface
* report generation
* progress tracking

Create and activate the main environment:

```bash
cd D:\novac
python -m venv venv
venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r backend\requirements.txt
```

If there is no complete requirements file yet, install the common backend packages:

```bash
pip install fastapi uvicorn python-multipart pymongo python-dotenv opencv-python numpy pillow pymupdf reportlab paddleocr paddlepaddle
```

Run a quick health check:

```bash
python -m compileall backend\app
```

Start the backend:

```bash
uvicorn backend.app.main:app --reload
```

or, if the app is run from inside the backend folder:

```bash
cd backend
uvicorn app.main:app --reload
```

## 2. MVSS Environment

MVSS is kept in a separate environment because it may require a specific PyTorch version that is different from the main backend.

Create the MVSS environment:

```bash
cd D:\novac
python -m venv mvss_venv
mvss_venv\Scripts\activate
```

Install MVSS dependencies according to the MVSS-Net requirements.

Example:

```bash
pip install -r backend\MVSS-Net\requirements.txt
```

If MVSS requires a specific older PyTorch version, keep that version inside `mvss_venv`.

Do not force the latest CUDA/PyTorch version unless MVSS has been tested with it.

NOVAC currently treats MVSS as CPU-safe by default to preserve compatibility and accuracy.

Expected MVSS behavior:

* MVSS model loads once in the worker.
* The loaded model is reused for later analysis.
* Repeated files use detector cache.
* If MVSS takes too long, it should timeout gracefully instead of freezing the full pipeline.

Check MVSS environment:

```bash
mvss_venv\Scripts\activate
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Run the MVSS runtime test if available:

```bash
python backend\scripts\test_mvss_cpu_runtime.py
```

## 3. TruFor Environment / Service

TruFor may run either through the main backend environment or through its own model directory depending on the current project setup.

The important requirement is that the TruFor root folder must be importable. The folder containing TruFor’s internal `lib/` directory should be added to the Python path by the service.

Expected TruFor behavior:

* TruFor model loads lazily.
* The loaded model is reused.
* Same-file detector results are cached using file hash.
* The service should not reload model weights on every request.

If TruFor import errors occur, check that the TruFor root path is correctly configured and that `lib/config.py` exists.

## 4. Frontend Setup

The frontend runs separately from the backend.

```bash
cd D:\novac\frontend
npm install
npm run dev
```

The frontend communicates with the backend API for:

* document upload
* analysis start
* progress polling
* result display
* PDF report download
* history page

## 5. Environment Variables

Create a `.env` file in the project/backend location expected by the app.

Example:

```env
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=novac
MVSS_DEVICE=cpu
MVSS_TIMEOUT_SECONDS=300
FULL_FORENSIC_MODE=true
```

Optional settings may include:

```env
DEBUG_DETECTOR_OUTPUTS=false
PARALLEL_DETECTORS=false
USE_FP16_INFERENCE=false
```

## 6. How the Multi-Environment Pipeline Works

The main FastAPI backend receives the uploaded file and controls the complete analysis flow.

Pipeline:

```txt
Frontend upload
↓
FastAPI backend receives file
↓
Backend creates case ID
↓
Progress status starts
↓
PDF text extraction / image preparation
↓
OCR
↓
Document quality check
↓
Document authenticity check
↓
ELA
↓
TruFor
↓
MVSS worker
↓
Text consistency
↓
Detector fusion
↓
Final risk calculation
↓
MongoDB save
↓
PDF report generation
↓
Frontend results page
```

The frontend does not directly call MVSS or TruFor. It only talks to the main backend.

The backend is responsible for calling each detector service. If a detector requires a separate environment, the backend service/runner uses that environment internally through a worker or subprocess.

For example:

```txt
Main backend venv
    ├── handles API, OCR, reports, MongoDB
    ├── calls TruFor service
    └── calls MVSS worker using mvss_venv
```

## 7. Progress Tracking

When analysis starts, the backend creates a case ID and updates the current progress stage.

The frontend polls:

```txt
GET /analysis/status/{case_id}
```

Example status response:

```json
{
  "case_id": "NOVAC-D03C6239",
  "stage": "Running MVSS model inference",
  "progress": 87,
  "message": "Running MVSS on CPU. This may take a while.",
  "complete": false,
  "error": null
}
```

Common stages:

```txt
Upload received
Preparing file
Extracting PDF text
Rendering PDF page
Running OCR
Checking document quality
Checking document authenticity
Running AI/synthetic detection
Running ELA analysis
Preparing TruFor input
Running TruFor model inference
Processing TruFor output
Preparing MVSS input
Running MVSS model inference
Processing MVSS output
Running text consistency analysis
Combining detector results
Calculating final risk
Saving result
Analysis complete
```

## 8. Detector Caching

Expensive detector outputs are cached using file hashes.

Cache keys include:

* file hash
* detector name
* model version
* config version

This means:

* First run of a new file may be slow.
* Second run after model warm-up should be faster.
* Re-analyzing the exact same file should be much faster due to cache hits.

## 9. Runtime Checks

To verify the main backend environment:

```bash
venv\Scripts\activate
python backend\scripts\check_runtime_health.py
```

To verify MVSS separately:

```bash
mvss_venv\Scripts\activate
python backend\scripts\test_mvss_cpu_runtime.py
```

To compile-check the backend:

```bash
python -m compileall backend\app backend\scripts
```

## 10. Git Ignore Notes

Do not commit virtual environments, uploads, caches, or model outputs.

The `.gitignore` should include:

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
```

Large model weights/checkpoints should also not be committed unless the repository is intentionally configured for them.

## 11. Running the Full Project

Terminal 1 — Backend:

```bash
cd D:\novac
venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

Terminal 2 — Frontend:

```bash
cd D:\novac\frontend
npm run dev
```

MongoDB should be running before starting analysis.

## 12. Notes

* The main backend environment controls the app.
* MVSS uses its own environment to avoid dependency conflicts.
* MVSS currently runs CPU-safe by default for compatibility.
* TruFor and MVSS use persistent model loading to reduce repeated startup cost.
* Detector caching improves repeated analysis speed.
* The generated PDF report is an automated investigation aid and should be manually reviewed before final decisions.


## Notes

* MVSS can run in a separate environment if its dependencies require isolation.
* Heavy forensic models may take time on first run because model weights need to load.
* Cached detector results and persistent model workers are used to improve repeated analysis speed.
* The generated report is an automated investigation aid and should not be treated as legal proof of fraud without manual verification.

## Disclaimer

NOVAC provides automated document risk analysis based on visual, textual, and forensic signals. Results should be reviewed by a human before making acceptance, rejection, or escalation decisions.
