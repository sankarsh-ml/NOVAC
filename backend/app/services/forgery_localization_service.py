import json
import os
import subprocess
import time
from pathlib import Path


def _timeout_seconds(default=180):

    try:
        return int(
            os.getenv(
                "TRUFOR_TIMEOUT_SECONDS",
                str(default)
            )
        )
    except Exception:
        return default


def _fallback(error, elapsed_time_seconds=0):

    return {
        "model_available": False,
        "model": "TruFor",
        "manipulation_detected": False,
        "forgery_score": 0,
        "confidence": 0,
        "suspicious_regions": [],
        "localization_map_path": None,
        "reasons": [error] if error else [],
        "model_error": error,
        "elapsed_time_seconds": round(
            float(elapsed_time_seconds or 0),
            3
        )
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
    timeout_seconds: int = None
) -> dict:

    started_at = time.perf_counter()
    timeout_seconds = timeout_seconds or _timeout_seconds()
    backend_dir = _backend_dir()
    runner = backend_dir / "app" / "services" / "forgery_localization_runner.py"
    image_path = Path(image_path).resolve()

    if not runner.exists():
        return _fallback(
            f"Forgery localization runner missing: {runner}",
            time.perf_counter() - started_at
        )

    python_exe = _venv_python()

    if not python_exe:
        return _fallback(
            "Forgery localization venv not found. Create it with "
            "backend\\scripts\\setup_forgery_model.bat. Expected "
            "backend\\model_venvs\\forgery_venv\\Scripts\\python.exe on Windows "
            "or backend/model_venvs/forgery_venv/bin/python on Linux/macOS.",
            time.perf_counter() - started_at
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
            cwd=str(backend_dir),
            env={
                **os.environ,
                "TRUFOR_TIMEOUT_SECONDS": str(timeout_seconds)
            },
            timeout=timeout_seconds + 10
        )

    except subprocess.TimeoutExpired:
        message = f"TruFor inference timed out after {timeout_seconds} seconds"
        return _fallback(
            message,
            timeout_seconds
        )

    except Exception as exc:
        return _fallback(
            f"Forgery localization failed to start: {exc}",
            time.perf_counter() - started_at
        )

    stdout = (completed.stdout or "").strip()

    if completed.returncode != 0 and not stdout:
        return _fallback(
            (completed.stderr or "Forgery localization runner failed").strip(),
            time.perf_counter() - started_at
        )

    try:
        result = json.loads(stdout)

    except Exception:
        return _fallback(
            (
                "Forgery localization runner returned invalid JSON. "
                f"stderr: {(completed.stderr or '').strip()[:500]}"
            ),
            time.perf_counter() - started_at
        )

    expected_defaults = _fallback(None)
    expected_defaults.update(result)
    expected_defaults["elapsed_time_seconds"] = round(
        float(
            expected_defaults.get("elapsed_time_seconds")
            or (time.perf_counter() - started_at)
        ),
        3
    )
    return expected_defaults
