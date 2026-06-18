# Phase 2 Field Extraction Model

Standalone field extraction pathway for `logasanjeev/indian-id-validator`.

This module is intentionally isolated from the main backend virtual environment. Do not install these dependencies into the main backend venv.

## Windows Setup

```powershell
cd backend
python -m venv field_extraction_venv
field_extraction_venv\Scripts\activate
pip install -r field_extraction_model/requirements.txt
```

## Linux/macOS Setup

```bash
cd backend
python -m venv field_extraction_venv
source field_extraction_venv/bin/activate
pip install -r field_extraction_model/requirements.txt
```

## Model Weights

Place the required model files here:

```text
backend/field_extraction_model/models/
```

Required files:

```text
Id_Classifier.pt
Aadhaar_Card.pt
Pan_Card.pt
Passport.pt
Voter_Id.pt
Driving_License.pt
```

## Single File Extraction

Windows:

```powershell
cd backend
field_extraction_venv\Scripts\python.exe field_extraction_model\run_extraction.py --input field_extraction_model\test_inputs\sample1.jpg --output field_extraction_model\test_outputs\sample1_result.json
```

Linux/macOS:

```bash
cd backend
field_extraction_venv/bin/python field_extraction_model/run_extraction.py --input field_extraction_model/test_inputs/sample1.jpg --output field_extraction_model/test_outputs/sample1_result.json
```

## Batch Test Runner

Windows:

```powershell
cd backend
field_extraction_venv\Scripts\python.exe field_extraction_model\test_extraction.py
```

Linux/macOS:

```bash
cd backend
field_extraction_venv/bin/python field_extraction_model/test_extraction.py
```

The test runner reads supported images and PDFs from `field_extraction_model/test_inputs/`, writes JSON to `field_extraction_model/test_outputs/`, and prints a short extraction summary.
