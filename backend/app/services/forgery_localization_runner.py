import argparse
import json
import sys
from pathlib import Path


def _fallback(error):

    return {
        "model_available": False,
        "model": "unavailable",
        "manipulation_detected": False,
        "forgery_score": 0,
        "confidence": 0,
        "suspicious_regions": [],
        "localization_map_path": None,
        "reasons": [],
        "model_error": error
    }


def main():

    parser = argparse.ArgumentParser(
        description="NOVAC optional forgery localization runner"
    )
    parser.add_argument(
        "--image",
        required=True
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(
            json.dumps(
                _fallback(f"Image not found: {image_path}")
            )
        )
        return

    backend_dir = Path(__file__).resolve().parents[2]
    model_dir = backend_dir / "models" / "forgery" / "trufor"
    checkpoint_candidates = [
        model_dir / "checkpoint.pth",
        model_dir / "ckpt.pth",
        model_dir / "trufor.pth"
    ]

    if not model_dir.exists() or not any(path.exists() for path in checkpoint_candidates):
        print(
            json.dumps(
                _fallback(
                    "TruFor assets are not installed. Run backend\\scripts\\setup_forgery_model.bat "
                    "to create the isolated venv, then place the TruFor repo/checkpoint under "
                    "backend\\models\\forgery\\trufor."
                )
            )
        )
        return

    print(
        json.dumps(
            _fallback(
                "TruFor assets were found, but model inference is not wired yet. "
                "Add the TruFor adapter inside this isolated runner before enabling results."
            )
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            json.dumps(
                _fallback(str(exc))
            )
        )
        print(
            f"Forgery localization runner error: {exc}",
            file=sys.stderr
        )
