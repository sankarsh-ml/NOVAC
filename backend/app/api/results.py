from fastapi import (
    APIRouter,
    HTTPException
)

from app.services.result_service import (

    get_all_results,

    get_result_by_case_id,

    delete_result,

    delete_all_results

)

router = APIRouter()


@router.get("/results")
def get_results():

    return get_all_results()


@router.get("/results/case/{analysis_id}")
def get_result(
    analysis_id: str
):

    result = get_result_by_case_id(
        analysis_id
    )

    if result is None:

        raise HTTPException(

            status_code=404,

            detail="Analysis not found"

        )

    return result

@router.delete("/results/case/{case_id}")
def delete_case(case_id: str):

    success = delete_result(case_id)

    if not success:

        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    return {
        "message": f"{case_id} deleted successfully"
    }

@router.delete("/results")
def delete_all():

    deleted_count = delete_all_results()

    return {
        "message": "All records deleted",
        "deleted_count": deleted_count
    }