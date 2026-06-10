# Optional Forgery Localization Model

NOVAC calls forgery localization through a separate virtual environment:

`backend/model_venvs/forgery_venv`

The FastAPI backend never imports heavy model libraries directly. It calls
`backend/app/services/forgery_localization_runner.py` with subprocess and parses
JSON from stdout.

TruFor is the preferred future model. To enable it, create the venv with:

`backend/scripts/setup_forgery_model.bat`

Then place the TruFor repo and checkpoint under:

`backend/models/forgery/trufor`

Accepted checkpoint filenames:

- `checkpoint.pth`
- `ckpt.pth`
- `trufor.pth`

Until the TruFor adapter is wired in the runner, NOVAC returns
`model_available: false` and continues with MVSS, OCR, ELA, masking, metadata,
and text consistency.
