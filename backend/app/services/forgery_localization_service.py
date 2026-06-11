import json
import subprocess
from pathlib import Path


def _fallback(error):

    return {
        "model_available": False,
        "model": "TruFor",
        "manipulation_detected": False,
        "forgery_score": 0,
        "confidence": 0,
        "suspicious_regions": [],
        "localization_map_path": None,
        "reasons": [],
        "model_error": error
    }


def _backend_dir():

    return Path(__file__).resolve().parents[2]


def _venv_python():

    backend_dir = _backend_dir()
    candidates = [
        backend_dir / "model_venvs" / "forgery_venv" / "Scripts" / "python.exe",
        backend_dir / "model_venvs" / "forgery_venv" / "bin" / "python",
        backend_dir / "forgery_venv" / "Scripts" / "python.exe",
        backend_dir / "forgery_venv" / "bin" / "python"
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def analyze_forgery_localization(
    image_path: str,
    timeout_seconds: int = 120
) -> dict:

    backend_dir = _backend_dir()
    runner = backend_dir / "app" / "services" / "forgery_localization_runner.py"

    if not runner.exists():
        return _fallback(
            f"Forgery localization runner missing: {runner}"
        )

    python_exe = _venv_python()

    if not python_exe:
        return _fallback(
            "Forgery localization venv not found. Create it with "
            "backend\\scripts\\setup_forgery_model.bat. Expected "
            "backend\\model_venvs\\forgery_venv\\Scripts\\python.exe on Windows "
            "or backend/model_venvs/forgery_venv/bin/python on Linux/macOS."
        )

    try:
        completed = subprocess.run(
            [
                str(python_exe),
                str(runner),
                "--image",
                str(image_path)
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

    except subprocess.TimeoutExpired:
        return _fallback(
            f"Forgery localization timed out after {timeout_seconds} seconds"
        )

    except Exception as exc:
        return _fallback(
            f"Forgery localization failed to start: {exc}"
        )

    stdout = (completed.stdout or "").strip()

    if completed.returncode != 0 and not stdout:
        return _fallback(
            (completed.stderr or "Forgery localization runner failed").strip()
        )

    try:
        result = json.loads(stdout)

    except Exception:
        return _fallback(
            (
                "Forgery localization runner returned invalid JSON. "
                f"stderr: {(completed.stderr or '').strip()[:500]}"
            )
        )

    expected_defaults = _fallback(None)
    expected_defaults.update(result)
    return expected_defaults
