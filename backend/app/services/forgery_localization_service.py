import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from app.services.detector_cache import (
    cache_key,
    detector_file_hash,
    get_cached_result,
    model_file_version,
    store_cached_result
)


logger = logging.getLogger(__name__)

_WORKER = None
_WORKER_LOCK = threading.Lock()
_WORKER_CALL_LOCK = threading.Lock()
_WORKER_QUEUE = None


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


def _max_inference_dimension(default=1600):

    try:
        return int(
            os.getenv(
                "TRUFOR_MAX_DIMENSION",
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
        ),
        "cache_hit": False
    }


def _image_size(image_path):
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return {
                "width": int(image.width),
                "height": int(image.height)
            }
    except Exception:
        return None


def _trufor_debug_context(image_path, substep=None):
    backend_dir = _backend_dir()
    repo_candidates = [
        backend_dir / "models" / "forgery" / "TruFor",
        backend_dir / "models" / "forgery" / "trufor"
    ]
    trufor_root = next((path for path in repo_candidates if path.exists()), None)
    checkpoint = _trufor_checkpoint()

    return {
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "sys_path_relevant": [
            item for item in sys.path
            if "novac" in item.lower() or "trufor" in item.lower() or "forgery" in item.lower()
        ][:12],
        "trufor_root_path": str(trufor_root) if trufor_root else None,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "image_path": str(image_path) if image_path else None,
        "image_size": _image_size(image_path) if image_path else None,
        "current_trufor_substep": substep
    }


def _log_trufor_failure(message, image_path, exc=None, substep=None):
    context = _trufor_debug_context(image_path, substep=substep)

    if exc is not None:
        logger.error(
            "%s | exception_type=%s message=%s traceback=%s context=%s",
            message,
            type(exc).__name__,
            exc,
            traceback.format_exc(),
            context
        )
    else:
        logger.error("%s | context=%s", message, context)


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


def _trufor_checkpoint():
    backend_dir = _backend_dir()
    candidates = [
        backend_dir / "models" / "forgery" / "checkpoints" / "trufor.pth.tar",
        backend_dir / "models" / "forgery" / "checkpoints" / "checkpoint.pth",
        backend_dir / "models" / "forgery" / "checkpoints" / "ckpt.pth",
        backend_dir / "models" / "forgery" / "TruFor" / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar",
        backend_dir / "models" / "forgery" / "trufor" / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar"
    ]

    return next((path for path in candidates if path.exists()), None)


def _trufor_cache_key(image_path, file_hash=None):
    checkpoint = _trufor_checkpoint()

    if not checkpoint:
        return None

    file_hash = file_hash or detector_file_hash(image_path)

    if not file_hash:
        return None

    return cache_key(
        "trufor",
        model_file_version(checkpoint),
        file_hash,
        f"maxdim={_max_inference_dimension()};fp16=false"
    )


def _reader_thread(process, output_queue):
    for line in process.stdout:
        try:
            output_queue.put(json.loads(line))
        except Exception:
            logger.warning("Ignoring non-JSON TruFor worker output: %s", line.strip())


def _start_worker():
    global _WORKER
    global _WORKER_QUEUE

    backend_dir = _backend_dir()
    runner = backend_dir / "app" / "services" / "forgery_localization_runner.py"
    python_exe = _venv_python()

    if not runner.exists():
        raise RuntimeError(f"Forgery localization runner missing: {runner}")

    if not python_exe:
        raise RuntimeError(
            "Forgery localization venv not found. Create it with "
            "backend\\scripts\\setup_forgery_model.bat."
        )

    process = subprocess.Popen(
        [
            str(python_exe),
            str(runner),
            "--worker"
        ],
        cwd=str(backend_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "TRUFOR_TIMEOUT_SECONDS": str(_timeout_seconds()),
            "PYTHONUNBUFFERED": "1"
        }
    )
    output_queue = queue.Queue()
    reader = threading.Thread(
        target=_reader_thread,
        args=(process, output_queue),
        daemon=True
    )
    reader.start()

    try:
        ready = output_queue.get(timeout=30)
    except queue.Empty as exc:
        process.kill()
        raise RuntimeError("TruFor worker did not become ready") from exc

    if not ready.get("ready"):
        process.kill()
        raise RuntimeError(f"TruFor worker failed to start: {ready}")

    _WORKER = process
    _WORKER_QUEUE = output_queue
    return process


def _get_worker():
    global _WORKER

    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.poll() is not None:
            _WORKER = _start_worker()

        return _WORKER


def _call_worker(image_path, file_hash=None, timeout_seconds=None):
    timeout_seconds = timeout_seconds or _timeout_seconds()

    with _WORKER_CALL_LOCK:
        logger.info("Using persistent TruFor worker")
        process = _get_worker()

        try:
            process.stdin.write(
                json.dumps({
                    "image_path": str(Path(image_path).resolve()),
                    "file_hash": file_hash
                }) + "\n"
            )
            process.stdin.flush()
            response = _WORKER_QUEUE.get(timeout=timeout_seconds + 10)
        except queue.Empty as exc:
            _log_trufor_failure(
                f"TruFor worker timed out after {timeout_seconds} seconds",
                image_path,
                exc=exc,
                substep="worker_response_wait"
            )
            with _WORKER_LOCK:
                if _WORKER is not None and _WORKER.poll() is None:
                    _WORKER.kill()
            raise TimeoutError(f"TruFor worker timed out after {timeout_seconds} seconds") from exc
        except Exception as exc:
            _log_trufor_failure(
                "TruFor worker call failed",
                image_path,
                exc=exc,
                substep="worker_call"
            )
            with _WORKER_LOCK:
                if _WORKER is not None and _WORKER.poll() is None:
                    _WORKER.kill()
            raise

        if not response.get("ok"):
            logger.error(
                "TruFor worker returned error: type=%s message=%s substep=%s traceback=%s context=%s",
                response.get("exception_type"),
                response.get("error"),
                response.get("current_trufor_substep"),
                response.get("traceback"),
                _trufor_debug_context(
                    image_path,
                    substep=response.get("current_trufor_substep")
                )
            )
            raise RuntimeError(response.get("error") or "TruFor worker failed")

        return response["result"]


def _run_one_shot(image_path, file_hash=None, timeout_seconds=None):
    timeout_seconds = timeout_seconds or _timeout_seconds()
    backend_dir = _backend_dir()
    runner = backend_dir / "app" / "services" / "forgery_localization_runner.py"
    python_exe = _venv_python()

    if not python_exe:
        raise RuntimeError(
            "Forgery localization venv not found. Create it with "
            "backend\\scripts\\setup_forgery_model.bat."
        )

    command = [
        str(python_exe),
        str(runner),
        "--image",
        str(Path(image_path).resolve())
    ]

    if file_hash:
        command.extend([
            "--file-hash",
            file_hash
        ])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(backend_dir),
        env={
            **os.environ,
            "TRUFOR_TIMEOUT_SECONDS": str(timeout_seconds)
        },
        timeout=timeout_seconds + 10
    )

    stdout = (completed.stdout or "").strip()

    if completed.returncode != 0 and not stdout:
        raise RuntimeError((completed.stderr or "Forgery localization runner failed").strip())

    return json.loads(stdout)


def analyze_forgery_localization(
    image_path: str,
    timeout_seconds: int = None,
    file_hash: str = None
) -> dict:

    started_at = time.perf_counter()
    timeout_seconds = timeout_seconds or _timeout_seconds()
    image_path = Path(image_path).resolve()
    file_hash = file_hash or detector_file_hash(image_path)
    result_cache_key = _trufor_cache_key(
        image_path,
        file_hash=file_hash
    )

    if result_cache_key:
        cached = get_cached_result(result_cache_key, "TruFor")

        if cached is not None:
            return cached

    try:
        result = _call_worker(
            image_path,
            file_hash=file_hash,
            timeout_seconds=timeout_seconds
        )

    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        message = f"TruFor inference timed out after {timeout_seconds} seconds"
        _log_trufor_failure(
            message,
            image_path,
            exc=exc,
            substep="timeout"
        )
        return _fallback(
            message,
            timeout_seconds
        )

    except Exception as exc:
        _log_trufor_failure(
            "TruFor persistent worker failed; falling back to one-shot subprocess",
            image_path,
            exc=exc,
            substep="persistent_worker"
        )

        try:
            result = _run_one_shot(
                image_path,
                file_hash=file_hash,
                timeout_seconds=timeout_seconds
            )
        except Exception as fallback_exc:
            _log_trufor_failure(
                "TruFor one-shot subprocess failed",
                image_path,
                exc=fallback_exc,
                substep="one_shot_subprocess"
            )
            return _fallback(
                f"Forgery localization failed to start: {fallback_exc}",
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
    expected_defaults["cache_hit"] = False

    if result_cache_key:
        store_cached_result(
            result_cache_key,
            expected_defaults
        )

    return expected_defaults


def stop_forgery_localization_worker():
    global _WORKER

    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.poll() is None:
            _WORKER.terminate()

        _WORKER = None
