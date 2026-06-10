from bson import ObjectId

from app.database.mongodb import (
    analysis_collection
)


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

    result = analysis_collection.delete_one(
        {"case_id": case_id}
    )

    return result.deleted_count > 0


def delete_all_results():

    result = analysis_collection.delete_many({})

    return result.deleted_count