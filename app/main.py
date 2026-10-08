"""FastAPI application entrypoint for the Swabi AI service."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.core.auth_context import set_user_id, set_user_jwt
from app.core.config import settings
from app.core.exceptions import (
    AIProviderError,
    AITimeoutError,
    AIValidationError,
    ConfigurationError,
    SwabiAPIError,
    SwabiAuthenticationError,
    ToolExecutionError,
)
from app.core.logging import request_id_var, setup_logging
from app.core.redis import redis_manager

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect Redis on startup and close on shutdown."""
    await redis_manager.connect()
    try:
        yield
    finally:
        await redis_manager.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router)


# Middleware


def _extract_bearer_token(auth_header: str) -> str | None:
    """Extract an opaque bearer token. Returns None on any non-Bearer scheme."""
    if not auth_header:
        return None
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = value.strip()
    return token or None


@app.middleware("http")
async def auth_context_middleware(request: Request, call_next):
    """Capture the caller's JWT and user id for the duration of the request.

    The token is intentionally not logged, not decoded, and not persisted.
    """
    auth_header = request.headers.get("Authorization", "")
    token = _extract_bearer_token(auth_header)

    set_user_jwt(token)
    set_user_id(request.headers.get("X-User-Id"))

    return await call_next(request)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a request ID to every request and propagate it in the response."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# Centralized exception handlers


@app.exception_handler(AITimeoutError)
async def _ai_timeout_handler(_request: Request, exc: AITimeoutError):
    return JSONResponse(status_code=504, content={"error": "ai_timeout", "detail": str(exc)})


@app.exception_handler(AIProviderError)
async def _ai_provider_handler(_request: Request, exc: AIProviderError):
    return JSONResponse(status_code=502, content={"error": "ai_provider", "detail": str(exc)})


@app.exception_handler(AIValidationError)
async def _ai_validation_handler(_request: Request, exc: AIValidationError):
    return JSONResponse(status_code=422, content={"error": "ai_validation", "detail": str(exc)})


@app.exception_handler(SwabiAuthenticationError)
async def _swabi_auth_handler(_request: Request, exc: SwabiAuthenticationError):
    return JSONResponse(status_code=401, content={"error": "swabi_auth", "detail": str(exc)})


@app.exception_handler(SwabiAPIError)
async def _swabi_api_handler(_request: Request, exc: SwabiAPIError):
    return JSONResponse(status_code=502, content={"error": "swabi_api", "detail": str(exc)})


@app.exception_handler(ToolExecutionError)
async def _tool_handler(_request: Request, exc: ToolExecutionError):
    return JSONResponse(status_code=500, content={"error": "tool", "detail": str(exc)})


@app.exception_handler(ConfigurationError)
async def _config_handler(_request: Request, exc: ConfigurationError):
    return JSONResponse(status_code=500, content={"error": "configuration", "detail": str(exc)})


@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.app_name, "version": settings.app_version}
