import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

cv2 = None
np = None


def _result(
    model_available=False,
    manipulation_detected=False,
    forgery_score=0,
    confidence=0,
    suspicious_regions=None,
    localization_map_path=None,
    reasons=None,
    model_error=None
):

    return {
        "model_available": model_available,
        "model": "TruFor",
        "manipulation_detected": manipulation_detected,
        "forgery_score": forgery_score,
        "confidence": confidence,
        "suspicious_regions": suspicious_regions or [],
        "localization_map_path": localization_map_path,
        "reasons": reasons or [],
        "model_error": model_error
    }


def _backend_dir():

    return Path(__file__).resolve().parents[2]


def _load_runtime_dependencies():

    global cv2
    global np

    missing = []

    if cv2 is None:
        try:
            import cv2 as cv2_module

            cv2 = cv2_module

        except Exception as exc:
            missing.append(f"opencv-python ({exc})")

    if np is None:
        try:
            import numpy as np_module

            np = np_module

        except Exception as exc:
            missing.append(f"numpy ({exc})")

    if missing:
        return (
            False,
            "TruFor runner dependencies missing: "
            + ", ".join(missing)
            + ". Run backend\\scripts\\setup_forgery_model.bat to create "
            "backend\\model_venvs\\forgery_venv."
        )

    return True, None


def _find_trufor_paths():

    backend_dir = _backend_dir()
    repo_candidates = [
        backend_dir / "models" / "forgery" / "TruFor",
        backend_dir / "models" / "forgery" / "trufor"
    ]
    checkpoint_candidates = [
        backend_dir / "models" / "forgery" / "checkpoints" / "trufor.pth.tar",
        backend_dir / "models" / "forgery" / "checkpoints" / "checkpoint.pth",
        backend_dir / "models" / "forgery" / "checkpoints" / "ckpt.pth",
        backend_dir / "models" / "forgery" / "TruFor" / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar",
        backend_dir / "models" / "forgery" / "trufor" / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar"
    ]

    repo_dir = next(
        (
            path
            for path in repo_candidates
            if (path / "TruFor_train_test" / "test.py").exists()
        ),
        None
    )
    checkpoint = next(
        (
            path
            for path in checkpoint_candidates
            if path.exists()
        ),
        None
    )

    return repo_dir, checkpoint


def _save_heatmap(map_array, output_path):

    normalized = np.clip(
        map_array,
        0,
        1
    )
    heatmap = np.uint8(
        normalized * 255
    )
    colored = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )
    cv2.imwrite(
        str(output_path),
        colored
    )


def _squeeze_map(array):

    if array is None:
        return None

    squeezed = np.asarray(array)

    while squeezed.ndim > 2 and 1 in squeezed.shape:
        squeezed = np.squeeze(squeezed)

    if squeezed.ndim == 3:
        squeezed = squeezed[0]

    if squeezed.ndim != 2:
        return None

    return squeezed.astype("float32")


def _first_npz_array(output, keys):

    for key in keys:
        if key in output.files:
            return output[key]

    return None


def _response_path(path, backend_dir):

    try:
        path = path.relative_to(backend_dir)

    except ValueError:
        pass

    return str(path).replace("\\", "/")


def _regions_from_map(map_array, confidence_map, image_shape, score):

    height, width = image_shape[:2]

    if map_array.shape[:2] != (height, width):
        map_array = cv2.resize(
            map_array,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    if confidence_map is not None and confidence_map.shape[:2] != (height, width):
        confidence_map = cv2.resize(
            confidence_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    image_area = float(width * height) if width and height else 1.0
    min_area = max(
        800,
        image_area * 0.0015
    )
    threshold = max(
        0.50,
        float(np.percentile(map_array, 95))
    )
    binary = np.uint8(
        map_array >= threshold
    ) * 255
    kernel = np.ones(
        (5, 5),
        np.uint8
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    regions = []

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = area / image_area

        if area < min_area:
            continue

        if area_ratio > 0.40 and score < 0.85:
            continue

        if w < 15 or h < 15:
            continue

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )
        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            thickness=-1
        )
        region_confidence = score

        if confidence_map is not None and np.any(mask > 0):
            region_confidence = float(
                np.mean(
                    confidence_map[mask > 0]
                )
            )

        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "area_ratio": round(area_ratio, 5),
            "type": "forgery_model",
            "reason": "TruFor detected possible manipulated region",
            "confidence": round(float(region_confidence), 3)
        })

    return sorted(
        regions,
        key=lambda item: item["area"],
        reverse=True
    )[:5]


def _run_trufor(image_path):

    dependencies_available, dependency_error = _load_runtime_dependencies()

    if not dependencies_available:
        return _result(
            model_error=dependency_error
        )

    backend_dir = _backend_dir()
    repo_dir, checkpoint = _find_trufor_paths()

    if not repo_dir:
        return _result(
            model_error=(
                "TruFor repository not found. Run backend\\scripts\\setup_forgery_model.bat "
                "and ensure backend\\models\\forgery\\TruFor\\TruFor_train_test\\test.py exists."
            )
        )

    if not checkpoint:
        return _result(
            model_error=(
                "TruFor checkpoint not found. Expected "
                "backend\\models\\forgery\\checkpoints\\trufor.pth.tar or "
                "backend\\models\\forgery\\TruFor\\TruFor_train_test\\pretrained_models\\trufor.pth.tar."
            )
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return _result(
            model_error=f"Cannot read image: {image_path}"
        )

    output_dir = backend_dir / "uploads" / "forgery_maps"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    token = uuid.uuid4().hex[:10]
    npz_path = output_dir / f"{Path(image_path).stem}_{token}_trufor.npz"
    map_path = output_dir / f"{Path(image_path).stem}_{token}_trufor_map.png"
    train_dir = repo_dir / "TruFor_train_test"
    test_script = train_dir / "test.py"

    command = [
        sys.executable,
        str(test_script),
        "-g",
        "-1",
        "-in",
        str(Path(image_path).resolve()),
        "-out",
        str(npz_path.resolve()),
        "-exp",
        "trufor_ph3",
        "TEST.MODEL_FILE",
        str(checkpoint.resolve())
    ]

    completed = subprocess.run(
        command,
        cwd=str(train_dir),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"
        },
        timeout=110
    )

    if completed.stdout:
        print(
            completed.stdout,
            file=sys.stderr
        )

    if completed.stderr:
        print(
            completed.stderr,
            file=sys.stderr
        )

    if completed.returncode != 0:
        return _result(
            model_error=(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "TruFor inference failed"
            )
        )

    if not npz_path.exists():
        return _result(
            model_error=f"TruFor did not produce expected output: {npz_path}"
        )

    output = np.load(
        str(npz_path)
    )
    map_array = _squeeze_map(
        _first_npz_array(
            output,
            [
                "map",
                "pred",
                "prediction",
                "localization",
                "localization_map",
                "anomaly_map"
            ]
        )
    )

    if map_array is None:
        return _result(
            model_error="TruFor output missing localization map"
        )

    score = float(
        np.asarray(output["score"]).mean()
        if "score" in output.files
        else np.mean(map_array)
    )
    confidence_map = _squeeze_map(
        _first_npz_array(
            output,
            [
                "conf",
                "confidence",
                "confidence_map"
            ]
        )
    )
    confidence = float(
        score
        if confidence_map is None
        else np.mean(confidence_map)
    )
    regions = _regions_from_map(
        map_array,
        confidence_map,
        image.shape,
        score
    )
    manipulation_detected = bool(
        score >= 0.55
        or regions
    )

    _save_heatmap(
        cv2.resize(
            map_array,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        ),
        map_path
    )

    reasons = []

    if manipulation_detected:
        reasons.append(
            "TruFor detected possible manipulated region"
        )

    return _result(
        model_available=True,
        manipulation_detected=manipulation_detected,
        forgery_score=round(score * 100, 2),
        confidence=round(confidence, 3),
        suspicious_regions=regions,
        localization_map_path=_response_path(
            map_path,
            backend_dir
        ),
        reasons=reasons,
        model_error=None
    )


def main():

    parser = argparse.ArgumentParser(
        description="NOVAC TruFor forgery localization runner"
    )
    parser.add_argument(
        "--image",
        required=True
    )
    args = parser.parse_args()
    image_path = Path(
        args.image
    )

    if not image_path.exists():
        print(
            json.dumps(
                _result(
                    model_error=f"Image not found: {image_path}"
                )
            )
        )
        return

    result = _run_trufor(
        image_path
    )
    print(
        json.dumps(result)
    )


if __name__ == "__main__":
    try:
        main()

    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                _result(
                    model_error="TruFor inference timed out"
                )
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                _result(
                    model_error=str(exc)
                )
            )
        )
        print(
            f"TruFor runner error: {exc}",
            file=sys.stderr
        )
