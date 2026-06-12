from datetime import datetime


analysis_status_store = {}


def _now():
    return datetime.utcnow().isoformat() + "Z"


def update_analysis_status(case_id, stage, progress, message, error=None):
    existing = analysis_status_store.get(case_id, {})
    started_at = existing.get("started_at") or _now()

    status = {
        "case_id": case_id,
        "stage": stage,
        "progress": int(max(0, min(progress, 100))),
        "message": message,
        "started_at": started_at,
        "updated_at": _now(),
        "error": error,
    }
    analysis_status_store[case_id] = status

    return status


def get_analysis_status(case_id):
    return analysis_status_store.get(case_id)
