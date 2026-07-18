"""FastAPI application entry point for PolicyLens AI."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1 import answers, documents, search
from app.core.exceptions import PolicyLensError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PolicyLens AI",
    description="Multi-model financial document intelligence assistant.",
    version="0.1.0",
)


@app.exception_handler(PolicyLensError)
async def policylens_error_handler(request: Request, exc: PolicyLensError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Deliberately generic: never surface the exception message, type, or
    # traceback (which could include internal file paths) to the client.
    logger.error("Unhandled error while processing request to %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", tags=["health"], summary="Health check")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(answers.router, prefix="/api/v1", tags=["answers"])
