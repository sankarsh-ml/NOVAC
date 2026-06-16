import argparse
import json
import os
import sys
import time
from pathlib import Path


def _backend_dir():
    return Path(__file__).resolve().parents[1]


def _default_image():
    backend_dir = _backend_dir()
    candidates = [
        backend_dir / "uploads" / "torn.jpg",
        backend_dir / "test_assets" / "torn.jpg",
        backend_dir.parent / "uploads" / "torn.jpg",
        backend_dir.parent / "test_assets" / "torn.jpg",
    ]

    return next((path for path in candidates if path.exists()), None)


def _print_json(label, value):
    print(f"{label}:")
    print(json.dumps(value, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Run only the NOVAC TruFor detector and print runtime diagnostics."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Image path to test. Defaults to uploads/torn.jpg or test_assets/torn.jpg if present."
    )
    args = parser.parse_args()

    backend_dir = _backend_dir()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    image_path = Path(args.image_path).resolve() if args.image_path else _default_image()

    if not image_path:
        raise SystemExit(
            "No image supplied and no torn sample found at backend/uploads/torn.jpg "
            "or backend/test_assets/torn.jpg. Usage: python backend/scripts/test_trufor_runtime.py path/to/image.jpg"
        )

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    from app.services.forgery_localization_service import analyze_forgery_localization

    env_info = {
        "cwd": os.getcwd(),
        "backend_dir": str(backend_dir),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "image_path": str(image_path),
        "trufor_timeout_seconds": os.getenv("TRUFOR_TIMEOUT_SECONDS", "180"),
        "trufor_max_dimension": os.getenv("TRUFOR_MAX_DIMENSION", "1600"),
    }
    _print_json("Environment", env_info)

    started_at = time.perf_counter()
    result = analyze_forgery_localization(str(image_path))
    wall_seconds = round(time.perf_counter() - started_at, 3)

    summary = {
        "wall_seconds": wall_seconds,
        "model_available": result.get("model_available"),
        "model": result.get("model"),
        "manipulation_detected": result.get("manipulation_detected"),
        "forgery_score": result.get("forgery_score"),
        "region_count": len(result.get("suspicious_regions", []) or []),
        "localization_map_path": result.get("localization_map_path"),
        "model_error": result.get("model_error"),
        "reasons": result.get("reasons", []),
    }
    _print_json("Timing Breakdown", result.get("timings", {}))
    _print_json("Result Summary", summary)

    if not result.get("model_available"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
