import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

cv2 = None
np = None
_TRUFOR_MODEL = None
_TRUFOR_DEVICE = None
_TRUFOR_MODEL_KEY = None
_TRUFOR_MODEL_LOCK = threading.Lock()
_TRUFOR_RESULT_CACHE = {}
_TRUFOR_RESULT_CACHE_LOCK = threading.Lock()
_TRUFOR_ROOT_LOGGED = False
_TRUFOR_CURRENT_SUBSTEP = "startup"


def _result(
    model_available=False,
    manipulation_detected=False,
    forgery_score=0,
    confidence=0,
    suspicious_regions=None,
    localization_map_path=None,
    reasons=None,
    model_error=None,
    elapsed_time_seconds=0,
    timings=None
):

    result = {
        "model_available": model_available,
        "model": "TruFor",
        "manipulation_detected": manipulation_detected,
        "forgery_score": forgery_score,
        "confidence": confidence,
        "suspicious_regions": suspicious_regions or [],
        "localization_map_path": localization_map_path,
        "reasons": reasons or [],
        "model_error": model_error,
        "elapsed_time_seconds": round(
            float(elapsed_time_seconds or 0),
            3
        )
    }

    if timings is not None:
        result["timings"] = timings

    return result


def _set_substep(name):
    global _TRUFOR_CURRENT_SUBSTEP
    _TRUFOR_CURRENT_SUBSTEP = name
    print(f"TruFor substep: {name}", file=sys.stderr)


def _image_size(image_path):
    try:
        if cv2 is not None:
            image = cv2.imread(str(image_path))
            if image is not None:
                height, width = image.shape[:2]
                return {
                    "width": int(width),
                    "height": int(height)
                }
    except Exception:
        pass

    return None


def _debug_context(image_path=None, repo_dir=None, checkpoint=None):
    return {
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "sys_path_relevant": [
            item for item in sys.path
            if "forgery" in item.lower() or "trufor" in item.lower() or "novac" in item.lower()
        ][:12],
        "trufor_root_path": str(repo_dir) if repo_dir else None,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "image_path": str(image_path) if image_path else None,
        "image_size": _image_size(image_path) if image_path else None,
        "current_trufor_substep": _TRUFOR_CURRENT_SUBSTEP
    }


def _print_exception_context(exc, image_path=None, repo_dir=None, checkpoint=None):
    print(f"TruFor exception type: {type(exc).__name__}", file=sys.stderr)
    print(f"TruFor exception message: {exc}", file=sys.stderr)
    print("TruFor traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(
        "TruFor debug context: "
        + json.dumps(_debug_context(image_path, repo_dir, checkpoint), default=str),
        file=sys.stderr
    )


def _backend_dir():

    return Path(__file__).resolve().parents[2]


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


def _sha256_file(file_path):

    digest = hashlib.sha256()

    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _model_version(checkpoint):

    stat = checkpoint.stat()
    return f"{checkpoint.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _cache_key(image_path, checkpoint, file_hash=None):

    file_hash = file_hash or _sha256_file(image_path)
    return (
        f"trufor:{_model_version(checkpoint)}:"
        f"maxdim={_max_inference_dimension()}:fp16=false:{file_hash}"
    )


def _cached_result(key):

    with _TRUFOR_RESULT_CACHE_LOCK:
        result = _TRUFOR_RESULT_CACHE.get(key)

    if result is None:
        return None

    print("Using cached TruFor result", file=sys.stderr)
    cached = copy.deepcopy(result)
    cached["cache_hit"] = True
    return cached


def _store_result(key, result):

    if not key or not isinstance(result, dict):
        return

    cached = copy.deepcopy(result)
    cached["cache_hit"] = False

    with _TRUFOR_RESULT_CACHE_LOCK:
        _TRUFOR_RESULT_CACHE[key] = cached


def _trufor_device(torch_module):

    if torch_module.cuda.is_available():
        return torch_module.device("cuda:0"), [0]

    return torch_module.device("cpu"), [-1]


def _resolve_trufor_import_root(repo_dir):

    candidates = [
        Path(repo_dir) / "TruFor_train_test",
        Path(repo_dir)
    ]

    for candidate in candidates:
        lib_config_package = candidate / "lib" / "config" / "__init__.py"
        lib_config_file = candidate / "lib" / "config.py"

        if lib_config_package.exists() or lib_config_file.exists():
            return candidate.resolve()

    return (Path(repo_dir) / "TruFor_train_test").resolve()


def _load_trufor_imports(repo_dir):

    global _TRUFOR_ROOT_LOGGED

    trufor_root = _resolve_trufor_import_root(repo_dir)
    lib_config_package = trufor_root / "lib" / "config" / "__init__.py"
    lib_config_file = trufor_root / "lib" / "config.py"

    if not _TRUFOR_ROOT_LOGGED:
        print(f"Resolved TruFor root: {trufor_root}", file=sys.stderr)
        print(
            "TruFor lib/config.py exists: "
            f"{str(lib_config_file.exists() or lib_config_package.exists()).lower()}",
            file=sys.stderr
        )
        print(
            f"TruFor lib/config package path: {lib_config_package}",
            file=sys.stderr
        )
        _TRUFOR_ROOT_LOGGED = True

    trufor_root_str = str(trufor_root)

    if trufor_root_str not in sys.path:
        sys.path.insert(0, trufor_root_str)

    import torch
    from torch.nn import functional as F
    from PIL import Image
    from lib.config import config as base_config
    from lib.config import update_config
    from lib.utils import get_model

    return torch, F, Image, base_config, update_config, get_model


def _torch_load_trusted_checkpoint(torch_module, checkpoint, device):

    try:
        return torch_module.load(
            str(checkpoint.resolve()),
            map_location=device,
            weights_only=False
        )

    except TypeError:
        return torch_module.load(
            str(checkpoint.resolve()),
            map_location=device
        )


def _get_trufor_model(repo_dir, train_dir, checkpoint, timings=None):

    global _TRUFOR_MODEL
    global _TRUFOR_DEVICE
    global _TRUFOR_MODEL_KEY

    import_started_at = time.perf_counter()
    torch, _, _, base_config, update_config, get_model = _load_trufor_imports(
        repo_dir
    )
    if timings is not None:
        timings["trufor_import_setup_seconds"] = round(
            time.perf_counter() - import_started_at,
            3
        )
    device, gpu_arg = _trufor_device(torch)
    model_key = (
        str(checkpoint.resolve()),
        str(device)
    )

    if _TRUFOR_MODEL is not None and _TRUFOR_MODEL_KEY == model_key:
        print("Using cached TruFor model", file=sys.stderr)
        if timings is not None:
            timings["trufor_model_load_seconds"] = 0
        return _TRUFOR_MODEL, _TRUFOR_DEVICE, torch

    with _TRUFOR_MODEL_LOCK:
        if _TRUFOR_MODEL is not None and _TRUFOR_MODEL_KEY == model_key:
            print("Using cached TruFor model", file=sys.stderr)
            if timings is not None:
                timings["trufor_model_load_seconds"] = 0
            return _TRUFOR_MODEL, _TRUFOR_DEVICE, torch

        print("Loading TruFor model...", file=sys.stderr)
        _set_substep("model_load")
        started_at = time.perf_counter()
        print(f"TruFor running on {device.type}", file=sys.stderr)

        if device.type == "cuda":
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = False
            cudnn.deterministic = False
            cudnn.enabled = True

        old_cwd = os.getcwd()
        os.chdir(str(train_dir))

        try:
            config = base_config.clone()
            args = SimpleNamespace(
                experiment="trufor_ph3",
                gpu=gpu_arg,
                opts=[
                    "TEST.MODEL_FILE",
                    str(checkpoint.resolve())
                ]
            )
            update_config(config, args)
            checkpoint_data = _torch_load_trusted_checkpoint(
                torch,
                checkpoint,
                device
            )
            model = get_model(config)
            model.load_state_dict(checkpoint_data["state_dict"])
            model = model.to(device)
            model.eval()

        finally:
            os.chdir(old_cwd)

        _TRUFOR_MODEL = model
        _TRUFOR_DEVICE = device
        _TRUFOR_MODEL_KEY = model_key
        print(
            f"TruFor model loaded in {time.perf_counter() - started_at:.3f} seconds",
            file=sys.stderr
        )
        if timings is not None:
            timings["trufor_model_load_seconds"] = round(
                time.perf_counter() - started_at,
                3
            )
        return _TRUFOR_MODEL, _TRUFOR_DEVICE, torch


def _run_trufor_model_to_npz(
    repo_dir,
    train_dir,
    checkpoint,
    model_image_path,
    npz_path,
    timings=None
):

    timings = timings if timings is not None else {}
    _set_substep("import_setup")
    torch, F, Image, _, _, _ = _load_trufor_imports(repo_dir)
    timings.setdefault("trufor_import_setup_seconds", 0)
    _set_substep("model_load")
    model, device, _ = _get_trufor_model(
        repo_dir,
        train_dir,
        checkpoint,
        timings=timings
    )
    _set_substep("image_preprocessing")
    preprocess_started_at = time.perf_counter()

    with Image.open(model_image_path).convert("RGB") as image:
        rgb_array = np.array(image)

    rgb = torch.tensor(
        rgb_array.transpose(2, 0, 1),
        dtype=torch.float
    ) / 256.0
    rgb = rgb.unsqueeze(0).to(device)
    preprocess_seconds = time.perf_counter() - preprocess_started_at
    _set_substep("inference")
    inference_started_at = time.perf_counter()

    with torch.inference_mode():
        pred, conf, det, npp = model(
            rgb,
            save_np=False
        )

    inference_seconds = time.perf_counter() - inference_started_at
    _set_substep("postprocessing")
    postprocess_started_at = time.perf_counter()

    out_dict = {}

    if conf is not None:
        conf = torch.squeeze(conf, 0)
        conf = torch.sigmoid(conf)[0]
        out_dict["conf"] = conf.cpu().numpy()

    if det is not None:
        out_dict["score"] = torch.sigmoid(det).item()

    pred = torch.squeeze(pred, 0)
    pred = F.softmax(pred, dim=0)[1]
    out_dict["map"] = pred.cpu().numpy()
    out_dict["imgsize"] = tuple(rgb.shape[2:])

    os.makedirs(
        os.path.dirname(str(npz_path)),
        exist_ok=True
    )
    np.savez(
        str(npz_path),
        **out_dict
    )

    return {
        "trufor_preprocess_seconds": round(preprocess_seconds, 3),
        "trufor_inference_seconds": round(inference_seconds, 3),
        "trufor_postprocess_seconds": round(
            time.perf_counter() - postprocess_started_at,
            3
        )
    }


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


def _run_trufor(image_path, file_hash=None):

    started_at = time.perf_counter()
    timings = {}
    _set_substep("runtime_dependency_import")
    setup_started_at = time.perf_counter()
    dependencies_available, dependency_error = _load_runtime_dependencies()
    timings["trufor_import_setup_seconds"] = round(
        time.perf_counter() - setup_started_at,
        3
    )

    if not dependencies_available:
        return _result(
            reasons=[dependency_error],
            model_error=dependency_error,
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
        )

    _set_substep("path_resolution")
    backend_dir = _backend_dir()
    repo_dir, checkpoint = _find_trufor_paths()

    if not repo_dir:
        return _result(
            reasons=[
                "TruFor repository not found"
            ],
            model_error=(
                "TruFor repository not found. Run backend\\scripts\\setup_forgery_model.bat "
                "and ensure backend\\models\\forgery\\TruFor\\TruFor_train_test\\test.py exists."
            ),
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
        )

    if not checkpoint:
        return _result(
            reasons=[
                "TruFor checkpoint not found"
            ],
            model_error=(
                "TruFor checkpoint not found. Expected "
                "backend\\models\\forgery\\checkpoints\\trufor.pth.tar or "
                "backend\\models\\forgery\\TruFor\\TruFor_train_test\\pretrained_models\\trufor.pth.tar."
            ),
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
        )

    result_cache_key = None

    try:
        result_cache_key = _cache_key(
            image_path,
            checkpoint,
            file_hash=file_hash
        )
        cached = _cached_result(result_cache_key)

        if cached is not None:
            return cached

    except Exception as exc:
        print(
            f"TruFor result cache unavailable: {exc}",
            file=sys.stderr
        )

    _set_substep("image_preprocessing")
    preprocess_started_at = time.perf_counter()
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return _result(
            reasons=[
                f"Cannot read image: {image_path}"
            ],
            model_error=f"Cannot read image: {image_path}",
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
        )

    output_dir = backend_dir / "uploads" / "forgery_maps"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    token = uuid.uuid4().hex[:10]
    npz_path = output_dir / f"{Path(image_path).stem}_{token}_trufor.npz"
    map_path = output_dir / f"{Path(image_path).stem}_{token}_trufor_map.png"
    model_image_path = Path(image_path)
    max_dimension = _max_inference_dimension()
    original_h, original_w = image.shape[:2]
    longest_side = max(original_w, original_h)

    if max_dimension > 0 and longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized = cv2.resize(
            image,
            (
                max(1, int(original_w * scale)),
                max(1, int(original_h * scale))
            ),
            interpolation=cv2.INTER_AREA
        )
        model_image_path = output_dir / f"{Path(image_path).stem}_{token}_trufor_input.png"
        cv2.imwrite(
            str(model_image_path),
            resized
        )

    train_dir = repo_dir / "TruFor_train_test"
    test_script = train_dir / "test.py"
    timings["trufor_preprocess_seconds"] = round(
        time.perf_counter() - preprocess_started_at,
        3
    )

    inference_started_at = time.perf_counter()
    inprocess_timings = None

    try:
        print("Using persistent TruFor worker", file=sys.stderr)
        inprocess_timings = _run_trufor_model_to_npz(
            repo_dir,
            train_dir,
            checkpoint,
            model_image_path,
            npz_path,
            timings=timings
        )
        timings.update(inprocess_timings)

    except Exception as exc:
        _print_exception_context(exc, image_path, repo_dir, checkpoint)
        print(
            f"TruFor persistent model path failed: {exc}",
            file=sys.stderr
        )
        traceback.print_exc(
            file=sys.stderr
        )
        print(
            "Falling back to test.py",
            file=sys.stderr
        )
        command = [
            sys.executable,
            str(test_script),
            "-g",
            "-1",
            "-in",
            str(model_image_path.resolve()),
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
            timeout=max(
                1,
                _timeout_seconds() - 5
            )
        )

        timings["trufor_inference_seconds"] = round(
            time.perf_counter() - inference_started_at,
            3
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
                reasons=[
                    "TruFor inference failed"
                ],
                model_error=(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "TruFor inference failed"
                ),
                elapsed_time_seconds=time.perf_counter() - started_at,
                timings=timings
            )

    if not npz_path.exists():
        return _result(
            reasons=[
                "TruFor did not produce expected localization output"
            ],
            model_error=f"TruFor did not produce expected output: {npz_path}",
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
        )

    _set_substep("postprocessing")
    postprocess_started_at = time.perf_counter()
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
            reasons=[
                "TruFor output missing localization map"
            ],
            model_error="TruFor output missing localization map",
            elapsed_time_seconds=time.perf_counter() - started_at,
            timings=timings
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

    _set_substep("annotation_generation")
    annotation_started_at = time.perf_counter()
    _save_heatmap(
        cv2.resize(
            map_array,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        ),
        map_path
    )
    timings["trufor_annotation_seconds"] = round(
        time.perf_counter() - annotation_started_at,
        3
    )

    reasons = []

    if manipulation_detected:
        reasons.append(
            "TruFor detected possible manipulated region"
        )

    timings["trufor_postprocess_seconds"] = round(
        time.perf_counter() - postprocess_started_at,
        3
    )
    timings["trufor_total_seconds"] = round(
        time.perf_counter() - started_at,
        3
    )
    print(
        f"TruFor preprocessing took {timings.get('trufor_preprocess_seconds', 0):.3f} seconds",
        file=sys.stderr
    )
    print(
        f"TruFor model inference took {timings.get('trufor_inference_seconds', 0):.3f} seconds",
        file=sys.stderr
    )
    print(
        f"TruFor postprocessing took {timings.get('trufor_postprocess_seconds', 0):.3f} seconds",
        file=sys.stderr
    )
    print(
        f"TruFor annotation generation took {timings.get('trufor_annotation_seconds', 0):.3f} seconds",
        file=sys.stderr
    )
    print(
        f"TruFor total took {timings.get('trufor_total_seconds', 0):.3f} seconds",
        file=sys.stderr
    )

    result = _result(
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
        model_error=None,
        elapsed_time_seconds=time.perf_counter() - started_at
    )
    result["timings"] = timings
    result["model_device"] = str(_TRUFOR_DEVICE.type) if _TRUFOR_DEVICE else "cpu"
    result["model_version"] = _model_version(checkpoint)
    result["cache_hit"] = False
    _store_result(
        result_cache_key,
        result
    )
    return result


def main():

    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="NOVAC TruFor forgery localization runner"
    )
    parser.add_argument(
        "--image",
        required=False
    )
    parser.add_argument(
        "--file-hash",
        required=False
    )
    parser.add_argument(
        "--worker",
        action="store_true"
    )
    args = parser.parse_args()

    if args.worker:
        print(
            json.dumps({
                "ready": True,
                "worker": "trufor"
            })
        )
        sys.stdout.flush()

        for line in sys.stdin:
            try:
                request = json.loads(line)
                image_path = Path(
                    request["image_path"]
                )

                with contextlib.redirect_stdout(sys.stderr):
                    result = _run_trufor(
                        image_path,
                        file_hash=request.get("file_hash")
                    )

                print(
                    json.dumps({
                        "ok": True,
                        "result": result
                    })
                )
                sys.stdout.flush()

            except Exception as exc:
                error_traceback = traceback.format_exc()
                print(
                    json.dumps({
                        "ok": False,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": error_traceback,
                        "current_trufor_substep": _TRUFOR_CURRENT_SUBSTEP
                    })
                )
                sys.stdout.flush()
                print(
                    f"TruFor worker error: {exc}",
                    file=sys.stderr
                )
                print(error_traceback, file=sys.stderr)

        return

    if not args.image:
        raise ValueError("--image is required unless --worker is used")

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        print(
            json.dumps(
                _result(
                    reasons=[
                        f"Image not found: {image_path}"
                    ],
                    model_error=f"Image not found: {image_path}",
                    elapsed_time_seconds=time.perf_counter() - started_at
                )
            )
        )
        return

    with contextlib.redirect_stdout(sys.stderr):
        result = _run_trufor(
            image_path,
            file_hash=args.file_hash
        )
    print(
        json.dumps(result)
    )


if __name__ == "__main__":
    try:
        main()

    except subprocess.TimeoutExpired:
        timeout_seconds = _timeout_seconds()
        message = f"TruFor inference timed out after {timeout_seconds} seconds"
        print(
            json.dumps(
                _result(
                    reasons=[message],
                    model_error=message,
                    elapsed_time_seconds=timeout_seconds,
                    timings={
                        "trufor_total_seconds": timeout_seconds,
                        "current_trufor_substep": _TRUFOR_CURRENT_SUBSTEP
                    }
                )
            )
        )

    except Exception as exc:
        error_traceback = traceback.format_exc()
        print(
            json.dumps(
                _result(
                    reasons=[str(exc)],
                    model_error=str(exc),
                    timings={
                        "current_trufor_substep": _TRUFOR_CURRENT_SUBSTEP,
                        "exception_type": type(exc).__name__,
                        "traceback": error_traceback
                    }
                )
            )
        )
        print(
            f"TruFor runner error: {exc}",
            file=sys.stderr
        )
        print(error_traceback, file=sys.stderr)
