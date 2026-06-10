from datetime import datetime

from app.database.mongodb import analysis_collection

import uuid


def save_analysis(result: dict):

    case_id = result.get(
        "case_id"
    ) or f"NOVAC-{uuid.uuid4().hex[:8].upper()}"

    document = {
        **result,
        "case_id": case_id,
        "timestamp": datetime.utcnow()
    }

    analysis_collection.insert_one(document)

    return case_id
