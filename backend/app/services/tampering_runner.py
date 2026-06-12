import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from app.services.detector_cache import (
    cache_key,
    detector_file_hash,
    get_cached_result,
    model_file_version,
    store_cached_result
)


logger = logging.getLogger(__name__)
MVSS_TIMEOUT_SECONDS = int(os.getenv("MVSS_TIMEOUT_SECONDS", "300"))
MVSS_DEVICE = os.getenv("MVSS_DEVICE", "cpu").lower()
MVSS_CONFIG_VERSION = "device=cpu;input=512;threshold=0.5;fp16=false;preprocess=qr-v1"

_WORKER = None
_WORKER_LOCK = threading.Lock()
_WORKER_CALL_LOCK = threading.Lock()
_WORKER_QUEUE = None
_WORKER_READER = None


def _backend_dir():
    return Path(__file__).resolve().parents[2]


def _project_root():
    return _backend_dir().parent


def _mvss_python():
    return _project_root() / "mvss_venv" / "Scripts" / "python.exe"


def _mvss_checkpoint():
    return _backend_dir() / "MVSS-Net" / "ckpt" / "mvssnet_casia.pt"


def _mvss_cache_key(image_path, file_hash=None):
    file_hash = file_hash or detector_file_hash(image_path)

    if not file_hash:
        return None

    return cache_key(
        "mvss",
        model_file_version(_mvss_checkpoint()),
        file_hash,
        MVSS_CONFIG_VERSION
    )


class MVSSWorkerTimeout(Exception):
    pass


def _mvss_inconclusive_result(error, timed_out=False, elapsed_seconds=0, cache_lookup_seconds=0):
    timings = {
        "mvss_total_seconds": round(float(elapsed_seconds or 0), 3),
        "mvss_preprocess_seconds": 0,
        "mvss_inference_seconds": 0,
        "mvss_postprocess_seconds": 0,
        "mvss_cache_lookup_seconds": round(float(cache_lookup_seconds or 0), 3),
        "mvss_cache_hit": False,
        "mvss_timed_out": bool(timed_out)
    }

    return {
        "enabled": True,
        "completed": False,
        "timed_out": bool(timed_out),
        "score": 0,
        "tampering_detected": False,
        "tampering_score": 0,
        "tampered_area_percent": 0,
        "mask_path": None,
        "mvss_confidence": 0,
        "raw_region_count": 0,
        "scoring_region_count": 0,
        "annotation_region_count": 0,
        "suspicious_region_count": 0,
        "suspicious_regions": [],
        "annotation_regions": [],
        "suppressed_regions": [],
        "suppressed_region_count": 0,
        "reasons": [
            "MVSS analysis was inconclusive due to timeout"
            if timed_out
            else "MVSS analysis was inconclusive"
        ],
        "error": error,
        "timings": timings,
        "model_device": "cpu",
        "model_version": model_file_version(_mvss_checkpoint()),
        "cache_hit": False
    }


def _reader_thread(process, output_queue):
    for line in process.stdout:
        try:
            output_queue.put(json.loads(line))
        except Exception:
            logger.warning("Ignoring non-JSON MVSS worker output: %s", line.strip())


def _start_worker():
    global _WORKER
    global _WORKER_QUEUE
    global _WORKER_READER

    python_exe = _mvss_python()
    worker_script = _backend_dir() / "app" / "services" / "mvss_worker.py"

    if not python_exe.exists():
        raise RuntimeError(f"MVSS venv Python not found: {python_exe}")

    process = subprocess.Popen(
        [
            str(python_exe),
            str(worker_script)
        ],
        cwd=str(_backend_dir()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "MVSS_DEVICE": "cpu",
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
        ready = output_queue.get(timeout=60)
    except queue.Empty as exc:
        process.kill()
        raise RuntimeError("MVSS worker did not become ready") from exc

    if not ready.get("ready"):
        process.kill()
        raise RuntimeError(f"MVSS worker failed to start: {ready}")

    _WORKER = process
    _WORKER_QUEUE = output_queue
    _WORKER_READER = reader
    return process


def _kill_worker():
    global _WORKER
    global _WORKER_QUEUE
    global _WORKER_READER

    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.poll() is None:
            _WORKER.kill()

        _WORKER = None
        _WORKER_QUEUE = None
        _WORKER_READER = None


def _get_worker():
    global _WORKER

    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.poll() is not None:
            _WORKER = _start_worker()

        return _WORKER


def _call_worker(image_path, timeout=None):
    timeout = timeout or MVSS_TIMEOUT_SECONDS

    with _WORKER_CALL_LOCK:
        process = _get_worker()

        try:
            process.stdin.write(json.dumps({"image_path": str(Path(image_path).resolve())}) + "\n")
            process.stdin.flush()
            response = _WORKER_QUEUE.get(timeout=timeout)
        except queue.Empty as exc:
            logger.warning("MVSS analysis timed out after %s seconds", timeout)
            _kill_worker()
            raise MVSSWorkerTimeout("MVSS analysis timed out") from exc

        except Exception:
            _kill_worker()
            raise

        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "MVSS worker failed")

        return response["result"]


def _run_one_shot(image_path):
    backend_dir = _backend_dir()
    mvss_python = _mvss_python()
    mvss_script = backend_dir / "MVSS-Net" / "mvss_predict.py"

    result = subprocess.run(
        [
            str(mvss_python),
            str(mvss_script),
            str(Path(image_path).resolve())
        ],
        capture_output=True,
        text=True,
        cwd=str(backend_dir),
        timeout=180
    )

    if result.returncode != 0:
        raise Exception(result.stderr)

    return json.loads(result.stdout)


def analyze_tampering(image_path, file_hash=None):
    started_at = time.perf_counter()
    cache_lookup_started_at = time.perf_counter()
    result_cache_key = _mvss_cache_key(image_path, file_hash=file_hash)
    cache_lookup_seconds = 0

    if result_cache_key:
        cached = get_cached_result(result_cache_key, "MVSS")
        cache_lookup_seconds = round(time.perf_counter() - cache_lookup_started_at, 3)

        if cached is not None:
            cached.setdefault("timings", {})
            cached["timings"]["mvss_cache_lookup_seconds"] = cache_lookup_seconds
            cached["timings"]["mvss_cache_hit"] = True
            cached["timings"]["mvss_timed_out"] = False
            cached["cache_hit"] = True
            return cached
    else:
        cache_lookup_seconds = round(time.perf_counter() - cache_lookup_started_at, 3)

    try:
        result = _call_worker(image_path)

    except MVSSWorkerTimeout:
        return _mvss_inconclusive_result(
            "MVSS analysis timed out",
            timed_out=True,
            elapsed_seconds=time.perf_counter() - started_at,
            cache_lookup_seconds=cache_lookup_seconds
        )

    except Exception as exc:
        if os.getenv("MVSS_ENABLE_ONESHOT_FALLBACK", "false").lower() == "true":
            logger.warning("MVSS persistent worker failed; falling back to one-shot subprocess: %s", exc)
            result = _run_one_shot(image_path)
        else:
            logger.exception("MVSS persistent worker failed")
            return _mvss_inconclusive_result(
                f"MVSS analysis failed: {exc}",
                timed_out=False,
                elapsed_seconds=time.perf_counter() - started_at,
                cache_lookup_seconds=cache_lookup_seconds
            )

    result.setdefault("timings", {})
    result["timings"].setdefault(
        "mvss_total_seconds",
        round(time.perf_counter() - started_at, 3)
    )
    result["timings"]["mvss_cache_lookup_seconds"] = cache_lookup_seconds
    result["timings"]["mvss_cache_hit"] = False
    result["timings"]["mvss_timed_out"] = False
    result["cache_hit"] = False
    result["model_device"] = "cpu"

    if result_cache_key:
        store_cached_result(result_cache_key, result)

    return result


def stop_tampering_worker():
    global _WORKER
    global _WORKER_QUEUE
    global _WORKER_READER

    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.poll() is None:
            _WORKER.terminate()

        _WORKER = None
        _WORKER_QUEUE = None
        _WORKER_READER = None
