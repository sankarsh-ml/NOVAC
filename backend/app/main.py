from fastapi import FastAPI
import logging

from app.api.upload import router as upload_router
from app.api.results import router as results_router

from fastapi.middleware.cors import CORSMiddleware

from app.api.report import router as report_router

from fastapi.staticfiles import StaticFiles

from app.services.forgery_localization_service import stop_forgery_localization_worker
from app.services.tampering_runner import stop_tampering_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


app = FastAPI(
    title="Government Fraud Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)

app.include_router(results_router)

app.include_router(report_router)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

@app.get("/")
def home():
    return {"message": "API Running"}


@app.on_event("shutdown")
def shutdown_workers():
    stop_forgery_localization_worker()
    stop_tampering_worker()
