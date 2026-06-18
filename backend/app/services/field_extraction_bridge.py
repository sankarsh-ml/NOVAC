from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
FIELD_EXTRACTION_DIR = BACKEND_DIR / "field_extraction_model"
FIELD_EXTRACTION_SCRIPT = FIELD_EXTRACTION_DIR / "run_extraction.py"
FIELD_EXTRACTION_VENV = BACKEND_DIR / "field_extraction_venv"
TIMEOUT_SECONDS = 180
LOG_TEXT_LIMIT = 2000


def _truncated(text: str | None, limit: int = LOG_TEXT_LIMIT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _python_executable() -> Path:
    if os.name == "nt":
        return FIELD_EXTRACTION_VENV / "Scripts" / "python.exe"
    return FIELD_EXTRACTION_VENV / "bin" / "python"


def _failed(input_path: str, error: str, **extra: Any) -> dict[str, Any]:
    result = {
        "status": "failed",
        "input_path": input_path,
        "error": error,
        "fields": {},
        "missing_fields": [],
        "warnings": [],
    }
    result.update(extra)
    return result


def _field_extraction_source_path(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_field_extraction_source.png")


def prepare_field_extraction_input(input_path: str | Path) -> Path:
    input_file = Path(input_path)

    if input_file.suffix.lower() != ".pdf":
        return input_file

    source_path = _field_extraction_source_path(input_file)
    if source_path.exists():
        return source_path

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF to prepare the field extraction source image."
        ) from exc

    document = fitz.open(str(input_file))
    try:
        if document.page_count == 0:
            raise ValueError("PDF has no pages")

        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(str(source_path))
    finally:
        document.close()

    return source_path


def run_field_extraction(input_path: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    python_exe = _python_executable()
    input_file = Path(input_path)
    extraction_input = input_file

    logger.info("Field extraction bridge starting input_path=%s", input_path)

    if not python_exe.exists():
        error = (
            "Field extraction environment not found. Please create "
            "backend/field_extraction_venv and install requirements."
        )
        logger.error("Field extraction bridge failed: %s", error)
        return _failed(input_path, error)

    if not FIELD_EXTRACTION_SCRIPT.exists():
        error = f"Field extraction script not found: {FIELD_EXTRACTION_SCRIPT}"
        logger.error("Field extraction bridge failed: %s", error)
        return _failed(input_path, error)

    if not input_file.exists():
        error = f"Uploaded file not found: {input_file}"
        logger.error("Field extraction bridge failed: %s", error)
        return _failed(input_path, error)

    try:
        extraction_input = prepare_field_extraction_input(input_file)
    except Exception as exc:
        logger.exception("Field extraction source preparation failed input_path=%s", input_path)
        return _failed(input_path, f"Field extraction source preparation failed: {exc}")

    output_handle = None
    output_path = None

    try:
        output_handle = tempfile.NamedTemporaryFile(
            prefix="novac_field_extraction_",
            suffix=".json",
            delete=False,
        )
        output_path = Path(output_handle.name)
        output_handle.close()

        command = [
            str(python_exe),
            str(FIELD_EXTRACTION_SCRIPT),
            "--input",
            str(extraction_input),
            "--output",
            str(output_path),
        ]
        redacted_command = [
            str(python_exe),
            str(FIELD_EXTRACTION_SCRIPT),
            "--input",
            "<uploaded-file>",
            "--output",
            "<temp-json>",
        ]
        logger.info(
            "Field extraction subprocess command=%s timeout=%ss",
            redacted_command,
            TIMEOUT_SECONDS,
        )

        completed = subprocess.run(
            command,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        stdout = _truncated(completed.stdout)
        stderr = _truncated(completed.stderr)

        if not output_path.exists():
            error = "Field extraction did not produce an output JSON file."
            logger.error(
                "Field extraction failed input_path=%s returncode=%s error=%s stderr=%s",
                extraction_input,
                completed.returncode,
                error,
                stderr,
            )
            return _failed(
                input_path,
                error,
                source_image_path=str(extraction_input.resolve()),
                subprocess_returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            error = f"Could not parse field extraction output JSON: {exc}"
            logger.exception("Field extraction output parse failed input_path=%s", input_path)
            return _failed(
                input_path,
                error,
                source_image_path=str(extraction_input.resolve()),
                subprocess_returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        if not isinstance(payload, dict):
            return _failed(
                input_path,
                "Field extraction output JSON was not an object.",
                source_image_path=str(extraction_input.resolve()),
                subprocess_returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        if completed.returncode != 0 and payload.get("status") != "failed":
            payload = {
                **payload,
                "status": "failed",
                "error": payload.get("error") or "Field extraction subprocess failed.",
            }

        if completed.returncode != 0:
            payload.setdefault("subprocess_returncode", completed.returncode)
            if stderr:
                payload.setdefault("stderr", stderr)

        payload["source_image_path"] = str(extraction_input.resolve())

        logger.info(
            "Field extraction bridge finished input_path=%s source_image_path=%s status=%s elapsed=%.3fs",
            input_path,
            extraction_input,
            payload.get("status"),
            time.perf_counter() - started_at,
        )
        return payload

    except subprocess.TimeoutExpired as exc:
        error = f"Field extraction timed out after {TIMEOUT_SECONDS} seconds."
        logger.error("Field extraction timeout input_path=%s", input_path)
        return _failed(
            input_path,
            error,
            stdout=_truncated(exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr=_truncated(exc.stderr if isinstance(exc.stderr, str) else ""),
        )

    except Exception as exc:
        logger.exception("Field extraction bridge crashed input_path=%s", input_path)
        return _failed(input_path, f"Field extraction bridge failed: {exc}")

    finally:
        if output_handle and not output_handle.file.closed:
            output_handle.close()
        if output_path:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to delete field extraction temp JSON: %s", output_path)
