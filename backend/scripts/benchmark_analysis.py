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
    print(f"  total_seconds: {timings.get('total_seconds', round(elapsed, 3))}s")
    print(f"  mvss_total_seconds: {timings.get('mvss_total_seconds', 'n/a')}s")
    print(f"  mvss_wall_seconds: {timings.get('mvss_wall_seconds', 'n/a')}s")
    print(f"  ocr_seconds: {timings.get('ocr_seconds', 'n/a')}s")
    print(f"  ela_seconds: {timings.get('ela_seconds', 'n/a')}s")
    print(f"  quality_seconds: {timings.get('quality_seconds', timings.get('document_quality_seconds', 'n/a'))}s")
    print(f"  authenticity_seconds: {timings.get('authenticity_seconds', 'n/a')}s")
    print(f"  text_consistency_seconds: {timings.get('text_consistency_seconds', 'n/a')}s")
    print(f"  wait_for_mvss_seconds: {timings.get('wait_for_mvss_seconds', 'n/a')}s")
    print(f"  trufor_total_seconds: {timings.get('trufor_total_seconds', trufor.get('elapsed_time_seconds', 'n/a'))}s")
    print(f"  trufor_seconds: {timings.get('trufor_seconds', timings.get('trufor_total_seconds', 'n/a'))}s")
    print(f"  mvss_inference_seconds: {timings.get('mvss_inference_seconds', 'n/a')}s")
    print(f"  mvss_cache_lookup_seconds: {timings.get('mvss_cache_lookup_seconds', 'n/a')}s")
    print(f"  mvss_timed_out: {timings.get('mvss_timed_out', mvss.get('timed_out', False))}")
    print(f"  save_seconds: {timings.get('save_seconds', 'n/a')}s")
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

    from app.api import upload as upload_module
    from app.services import detector_cache

    file_path = Path(args.file).resolve()
    second_file_path = Path(args.second_file).resolve() if args.second_file else file_path

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    if not second_file_path.exists():
        raise SystemExit(f"Second file not found: {second_file_path}")

    runs = [
        ("Old sequential mode", False, file_path),
        ("New MVSS-overlap mode", True, second_file_path)
    ]

    for label, parallel_mvss_pipeline, run_file in runs:
        with detector_cache._CACHE_LOCK:
            detector_cache._CACHE.clear()

        os.environ["PARALLEL_MVSS_PIPELINE"] = (
            "true"
            if parallel_mvss_pipeline
            else "false"
        )
        upload_module.PARALLEL_MVSS_PIPELINE = parallel_mvss_pipeline
        case_id = f"BENCH-{uuid.uuid4().hex[:8].upper()}"
        extension = run_file.suffix.lower()
        started_at = time.perf_counter()
        result = upload_module._run_saved_analysis(
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
