# TruFor Forgery Localization Setup

NOVAC runs TruFor through an isolated subprocess runner:

`FastAPI -> forgery_localization_service.py -> subprocess -> forgery_localization_runner.py`

The main backend virtual environment is not used for TruFor dependencies.

## Setup Command

From the project root, run:

```bat
backend\scripts\setup_forgery_model.bat
```

The setup script creates or reuses the isolated venv at:

```text
backend\model_venvs\forgery_venv
```

It clones the official repository to:

```text
backend\models\forgery\TruFor
```

and installs inference dependencies by calling only:

```bat
backend\model_venvs\forgery_venv\Scripts\python.exe -m pip ...
```

## Expected Folder Structure

```text
backend/
  app/
    services/
      forgery_localization_service.py
      forgery_localization_runner.py
  model_venvs/
    forgery_venv/
      Scripts/
        python.exe
  models/
    forgery/
      README.md
      checkpoints/
        trufor.pth.tar
      TruFor/
        TruFor_train_test/
          test.py
          lib/
            config/
              trufor_ph3.yaml
          pretrained_models/
            noiseprint++/
              noiseprint++.th
            segformers/
              mit_b2.pth
```

## TruFor Entrypoint

NOVAC uses the upstream inference command:

```bat
python test.py -g -1 -in path\to\image.png -out path\to\output.npz -exp trufor_ph3 TEST.MODEL_FILE path\to\trufor.pth.tar
```

The upstream `test.py` output is an `.npz` file. Documented keys are:

- `map`: anomaly localization map
- `conf`: confidence map
- `score`: image integrity score in `[0, 1]`
- `imgsize`: processed image size
- `np++`: optional Noiseprint++ output if `--save_np` is used

TruFor logs go to stderr. NOVAC runner stdout must remain a single JSON object.

## Checkpoint

NOVAC expects the final TruFor checkpoint at:

```text
backend\models\forgery\checkpoints\trufor.pth.tar
```

The setup script attempts to download:

```text
https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
```

If automatic download fails or times out:

1. Download `TruFor_weights.zip` from the URL above.
2. Unzip it.
3. Copy `trufor.pth.tar` to:

```text
backend\models\forgery\checkpoints\trufor.pth.tar
```

Until that file exists, NOVAC upload remains safe and the runner returns
`model_available: false`.

## Manual Runner Test

From the project root:

```bat
backend\model_venvs\forgery_venv\Scripts\python.exe backend\app\services\forgery_localization_runner.py --image backend\uploads\your_image.png
```

## Runtime Configuration

The FastAPI service and runner use these optional environment variables:

```text
TRUFOR_TIMEOUT_SECONDS=180
TRUFOR_MAX_DIMENSION=1600
```

`TRUFOR_TIMEOUT_SECONDS` controls how long upload waits for TruFor before
returning safe unavailable JSON. `TRUFOR_MAX_DIMENSION` bounds the image side
length sent to TruFor for practical upload latency; localization maps and
regions are still returned in the original image coordinate space.

## Success JSON

```json
{
  "model_available": true,
  "model": "TruFor",
  "manipulation_detected": true,
  "forgery_score": 72.4,
  "confidence": 0.81,
  "suspicious_regions": [],
  "localization_map_path": "backend/uploads/forgery_maps/example_trufor_map.png",
  "reasons": ["TruFor detected possible manipulated region"],
  "model_error": null
}
```

## Failure JSON

```json
{
  "model_available": false,
  "model": "TruFor",
  "manipulation_detected": false,
  "forgery_score": 0,
  "confidence": 0,
  "suspicious_regions": [],
  "localization_map_path": null,
  "reasons": [],
  "model_error": "specific setup/checkpoint/dependency error"
}
```
