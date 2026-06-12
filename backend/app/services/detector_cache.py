import copy
import hashlib
import logging
import os
import threading
from pathlib import Path


logger = logging.getLogger(__name__)

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def sha256_file(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def model_file_version(file_path):
    path = Path(file_path)

    if not path.exists():
        return "missing"

    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def cache_key(detector_name, model_version, file_hash, config_version="default"):
    return f"{detector_name}:{model_version}:{config_version}:{file_hash}"


def get_cached_result(key, detector_label):
    with _CACHE_LOCK:
        result = _CACHE.get(key)

    if result is None:
        return None

    logger.info("Using cached %s result", detector_label)
    cached = copy.deepcopy(result)
    cached["cache_hit"] = True
    return cached


def store_cached_result(key, result):
    if not isinstance(result, dict):
        return result

    cached = copy.deepcopy(result)
    cached["cache_hit"] = False

    with _CACHE_LOCK:
        _CACHE[key] = cached

    return result


def detector_file_hash(file_path):
    if not file_path or not os.path.exists(file_path):
        return None

    return sha256_file(file_path)
