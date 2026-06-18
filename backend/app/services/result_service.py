import logging
from pathlib import Path

from app.database.mongodb import (
    analysis_collection
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

LOCAL_ARTIFACT_ROOTS = {
    "uploads": (
        BACKEND_DIR / "uploads",
        PROJECT_DIR / "uploads"
    ),
    "reports": (
        BACKEND_DIR / "reports",
        PROJECT_DIR / "reports"
    )
}


def _allowed_roots():

    return tuple(
        root.resolve()
        for roots in LOCAL_ARTIFACT_ROOTS.values()
        for root in roots
    )


def _is_within_allowed_root(path):

    resolved_path = path.resolve()

    for root in _allowed_roots():
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue

    return False


def _candidate_artifact_paths(value):

    if not isinstance(value, str):
        return []

    raw_value = value.strip()

    if not raw_value:
        return []

    lowered = raw_value.lower()

    if (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("data:")
    ):
        return []

    normalized = raw_value.replace("\\", "/")

    if (
        normalized.startswith("/uploads/")
        or normalized.startswith("/reports/")
    ):
        normalized = normalized.lstrip("/")

    path = Path(normalized)

    if path.is_absolute():
        return [path]

    parts = path.parts

    if not parts:
        return []

    root_name = parts[0]
    roots = LOCAL_ARTIFACT_ROOTS.get(root_name)

    if not roots:
        return []

    return [
        root.joinpath(*parts[1:])
        for root in roots
    ]


def _walk_path_values(value):

    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and "path" in key.lower()
                and isinstance(child, str)
            ):
                yield child

            yield from _walk_path_values(child)

    elif isinstance(value, list):
        for child in value:
            yield from _walk_path_values(child)


def _collect_local_artifact_paths(document):

    paths = []

    for value in _walk_path_values(document):
        paths.extend(
            _candidate_artifact_paths(value)
        )

    stored_filename = document.get("stored_filename")

    if stored_filename:
        paths.extend(
            _candidate_artifact_paths(
                f"uploads/{stored_filename}"
            )
        )

    case_id = document.get("case_id")

    if case_id:
        paths.extend(
            _candidate_artifact_paths(
                f"reports/report_{case_id}.pdf"
            )
        )

    unique_paths = []
    seen = set()

    for path in paths:
        try:
            resolved_path = path.resolve()
        except OSError:
            continue

        if resolved_path in seen:
            continue

        if not _is_within_allowed_root(resolved_path):
            continue

        seen.add(resolved_path)
        unique_paths.append(resolved_path)

    return unique_paths


def _delete_local_artifacts(document):

    deleted_paths = []

    for path in _collect_local_artifact_paths(document):
        try:
            if path.exists() and (
                path.is_file()
                or path.is_symlink()
            ):
                path.unlink()
                deleted_paths.append(str(path))
        except OSError:
            logger.exception(
                "Unable to delete local artifact: %s",
                path
            )

    return deleted_paths


def get_all_results():

    results = []

    for doc in analysis_collection.find():

        results.append({

            "case_id":
                doc.get(
                    "case_id"),

            "filename":
                doc.get(
                    "filename",
                    "Unknown"
                ),

            "risk_level":
                doc.get(
                    "fraud_analysis",
                    {}
                ).get(
                    "risk_level",
                    "Unknown"
                ),

            "fraud_score":
                doc.get(
                    "fraud_analysis",
                    {}
                ).get(
                    "fraud_score",
                    0
                ),

            "field_extraction_status":
                doc.get(
                    "field_extraction",
                    {}
                ).get(
                    "status",
                    "not_run"
                )

        })

    return results


def get_result_by_case_id(case_id):

    document = analysis_collection.find_one(
        {"case_id": case_id}
    )

    if not document:
        return None

    document["_id"] = str(document["_id"])

    return document

def delete_result(case_id):

    document = analysis_collection.find_one(
        {"case_id": case_id}
    )

    if not document:
        return False

    result = analysis_collection.delete_one(
        {"case_id": case_id}
    )

    if result.deleted_count <= 0:
        return False

    _delete_local_artifacts(document)

    return True


def delete_all_results():

    documents = list(
        analysis_collection.find()
    )

    result = analysis_collection.delete_many({})

    if result.deleted_count > 0:
        for document in documents:
            _delete_local_artifacts(document)

    return result.deleted_count
