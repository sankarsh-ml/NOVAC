from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
TEST_INPUTS = MODULE_DIR / "test_inputs"
TEST_OUTPUTS = MODULE_DIR / "test_outputs"
RUNNER = MODULE_DIR / "run_extraction.py"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def _output_path(input_path: Path) -> Path:
    return TEST_OUTPUTS / f"{input_path.stem}_result.json"


def _summarize(input_path: Path, output_path: Path, return_code: int) -> str:
    if not output_path.exists():
        return f"{input_path.name} | status=failed | error=no output JSON | return_code={return_code}"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    status = payload.get("status", "unknown")
    document_type = payload.get("document_type", "")
    field_names = ", ".join(sorted((payload.get("fields") or {}).keys()))
    warnings = payload.get("warnings") or []
    error = payload.get("error")

    parts = [
        input_path.name,
        f"status={status}",
        f"document_type={document_type or 'n/a'}",
        f"fields=[{field_names}]",
    ]
    if warnings:
        parts.append(f"warnings={'; '.join(map(str, warnings))}")
    if error:
        parts.append(f"error={error}")
    return " | ".join(parts)


def main() -> int:
    TEST_OUTPUTS.mkdir(parents=True, exist_ok=True)
    inputs = sorted(path for path in TEST_INPUTS.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not inputs:
        print(f"No test inputs found in {TEST_INPUTS}")
        return 1

    overall_status = 0
    for input_path in inputs:
        output_path = _output_path(input_path)
        command = [
            sys.executable,
            str(RUNNER),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        result = subprocess.run(command, cwd=MODULE_DIR.parent, text=True, capture_output=True)
        if result.returncode != 0:
            overall_status = result.returncode
        print(_summarize(input_path, output_path, result.returncode))

    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
