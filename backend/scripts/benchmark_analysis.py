import argparse
import os
import sys
import time
import uuid
from pathlib import Path


def _repo_backend_dir():
    return Path(__file__).resolve().parents[1]


def _print_run(label, elapsed, result):
    timings = result.get("timings", {}) or {}
    trufor = result.get("forgery_localization_analysis", {}) or {}
    mvss = result.get("tampering_analysis", {}) or {}

    print(f"\n{label}")
    print(f"  total wall time: {elapsed:.3f}s")
    print(f"  total reported: {timings.get('total_seconds', 'n/a')}s")
    print(f"  OCR: {timings.get('ocr_seconds', 'n/a')}s")
    print(f"  TruFor: {timings.get('trufor_total_seconds', trufor.get('elapsed_time_seconds', 'n/a'))}s")
    print(f"  MVSS: {timings.get('mvss_total_seconds', 'n/a')}s")
    print(f"  MVSS inference: {timings.get('mvss_inference_seconds', 'n/a')}s")
    print(f"  MVSS cache lookup: {timings.get('mvss_cache_lookup_seconds', 'n/a')}s")
    print(f"  MVSS timed out: {timings.get('mvss_timed_out', mvss.get('timed_out', False))}")
    print(f"  ELA: {timings.get('ela_seconds', 'n/a')}s")
    print(f"  save: {timings.get('save_seconds', 'n/a')}s")
    print(f"  TruFor cache hit: {trufor.get('cache_hit', False)}")
    print(f"  MVSS cache hit: {mvss.get('cache_hit', False)}")
    print(f"  risk level: {(result.get('fraud_analysis') or {}).get('risk_level', 'n/a')}")
    print(f"  fraud score: {(result.get('fraud_analysis') or {}).get('fraud_score', 'n/a')}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NOVAC analysis cold and warm runs."
    )
    parser.add_argument(
        "file",
        help="Path to an image or PDF for cold and cache-hit runs."
    )
    parser.add_argument(
        "--second-file",
        help="Optional different file for warm-model/no-reload run."
    )
    args = parser.parse_args()

    backend_dir = _repo_backend_dir()
    os.chdir(str(backend_dir))

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.api.upload import _run_saved_analysis

    file_path = Path(args.file).resolve()
    second_file_path = Path(args.second_file).resolve() if args.second_file else file_path

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    if not second_file_path.exists():
        raise SystemExit(f"Second file not found: {second_file_path}")

    runs = [
        ("First run (cold load)", file_path),
        (
            "Second run (warm MVSS model)"
            if second_file_path != file_path
            else "Second run (same file, cache expected)",
            second_file_path
        ),
        ("Third run (same-file MVSS cache hit)", file_path)
    ]

    for label, run_file in runs:
        case_id = f"BENCH-{uuid.uuid4().hex[:8].upper()}"
        extension = run_file.suffix.lower()
        started_at = time.perf_counter()
        result = _run_saved_analysis(
            case_id,
            run_file.name,
            run_file.name,
            extension,
            str(run_file)
        )
        elapsed = time.perf_counter() - started_at
        _print_run(
            label,
            elapsed,
            result
        )


if __name__ == "__main__":
    main()
