"""RAG support route. Called by the frontend; user JWT is the only credential.

Frontend sends:
    Authorization: Bearer <user-jwt>

The AI service does NOT accept a service token on this route. The user
JWT is forwarded unchanged to the Swabi NestJS backend on any live-data
call inside the service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.capabilities.support.intent_llm import INTENT_PROMPT_VERSION
from app.capabilities.support.prompts import SUPPORT_PROMPT_VERSION
from app.capabilities.support.schemas import (
    AnswerType,
    SupportAskRequest,
    SupportAskResponse,
)
from app.capabilities.support.service import SupportAskPlan, ask, prepare_support_ask
from app.core.config import settings
from app.core.response_cache import (
    acquire_lock,
    get_json,
    make_cache_key,
    normalize_message,
    release_lock,
    set_json,
)
from app.core.route_auth import require_user_capability

router = APIRouter(prefix="/api/v1/support", tags=["support"])
logger = logging.getLogger(__name__)

# Register a Bearer security scheme rather than a plain header parameter so
# Swagger UI can use its Authorize dialog and include the token automatically.
user_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="UserBearerAuth",
    description="User JWT forwarded to the Swabi backend for live-data requests.",
)


def _support_cache_key(
    payload: SupportAskRequest,
    plan: SupportAskPlan,
) -> str:
    version = plan.version_status
    if not plan.cache_eligible or version is None:
        raise ValueError("A verified static knowledge-base plan is required")

    return make_cache_key(
        "support-static-rag:v1",
        {
            "question": normalize_message(payload.question),
            "audience": payload.audience.value,
            "knowledge_area": (
                payload.knowledge_area.value if payload.knowledge_area is not None else None
            ),
            "knowledge_version": version.local_version,
            "intent_prompt_version": INTENT_PROMPT_VERSION,
            "prompt_version": SUPPORT_PROMPT_VERSION,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "fallback_enabled": settings.llm_fallback_enabled,
            "fallback_model": settings.gemini_model,
            "response_schema_version": "v1",
            "retrieval": {
                "top_k": settings.rag_top_k,
                "min_score": settings.rag_min_score,
                "max_context_chunks": settings.support_max_context_chunks,
                "max_chunk_chars": settings.support_max_chunk_chars,
                "embedding_model": settings.rag_embedding_model,
                "reranker_enabled": settings.rag_reranker_enabled,
                "reranker_model": settings.rag_reranker_model,
                "answer_max_tokens": settings.support_answer_max_tokens,
            },
        },
    )


def _is_cacheable_static_response(
    response: SupportAskResponse,
    *,
    knowledge_version: str,
) -> bool:
    return (
        response.answer_type == AnswerType.ANSWERED
        and response.live_data_used is False
        and not response.live_data_sources
        and response.live_data_warning is None
        and response.retrieval_mode_used == "hybrid"
        and bool(response.citations)
        and response.knowledge_version == knowledge_version
        and response.prompt_version == SUPPORT_PROMPT_VERSION
    )


async def _cached_support_response(
    key: str,
    *,
    knowledge_version: str,
    request_id: str,
) -> SupportAskResponse | None:
    cached = await get_json(key)
    if cached is None:
        return None

    try:
        response = SupportAskResponse.model_validate(cached)
    except ValueError:
        logger.warning(
            "Invalid support response cache payload; treating as a miss",
            extra={"event": "support_response_cache_invalid"},
        )
        return None

    if not _is_cacheable_static_response(response, knowledge_version=knowledge_version):
        logger.warning(
            "Non-static support response found in cache; treating as a miss",
            extra={"event": "support_response_cache_unsafe_payload"},
        )
        return None

    response.request_id = request_id
    logger.info(
        "support response cache hit",
        extra={"event": "support_response_cache_hit"},
    )
    return response


@router.post("/ask", response_model=SupportAskResponse)
async def support_ask(
    payload: SupportAskRequest,
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(user_bearer),
    ],
    _principal: Annotated[
        object | None,
        require_user_capability("ai:support:ask", legacy_service_token=False),
    ] = None,
) -> SupportAskResponse:
    if not settings.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG support is not enabled",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_jwt = credentials.credentials.strip()
    if not user_jwt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In gateway mode the authoritative audience is in the verified principal
    # assertion.  The request body remains temporarily compatible but cannot
    # select a different knowledge scope.
    if (
        _principal is not None
        and getattr(_principal, "audience", payload.audience) != payload.audience
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requested audience is not authorized",
        )

    request_id = getattr(request.state, "request_id", "")
    plan: SupportAskPlan | None = None
    cache_key: str | None = None
    cache_knowledge_version: str | None = None
    lock_token: str | None = None

    if settings.enable_support_rag_response_cache:
        plan = await asyncio.to_thread(prepare_support_ask, payload)
        if plan.cache_eligible and plan.version_status is not None:
            cache_key = _support_cache_key(payload, plan)
            cache_knowledge_version = plan.version_status.local_version
            cached = await _cached_support_response(
                cache_key,
                knowledge_version=cache_knowledge_version,
                request_id=request_id,
            )
            if cached is not None:
                return cached

            logger.info(
                "support response cache miss",
                extra={"event": "support_response_cache_miss"},
            )
            redis_available, lock_token = await acquire_lock(
                cache_key,
                ttl_seconds=settings.support_rag_response_cache_lock_ttl_seconds,
            )
            if redis_available and lock_token is None:
                logger.info(
                    "support response cache lock busy; waiting for result",
                    extra={"event": "support_response_cache_lock_wait"},
                )
                lock_available = True
                deadline = (
                    asyncio.get_running_loop().time()
                    + settings.support_rag_response_cache_lock_wait_seconds
                )
                while asyncio.get_running_loop().time() < deadline:
                    cached = await _cached_support_response(
                        cache_key,
                        knowledge_version=cache_knowledge_version,
                        request_id=request_id,
                    )
                    if cached is not None:
                        return cached

                    lock_available, replacement_token = await acquire_lock(
                        cache_key,
                        ttl_seconds=settings.support_rag_response_cache_lock_ttl_seconds,
                    )
                    if replacement_token is not None:
                        lock_token = replacement_token
                        break
                    if not lock_available:
                        break
                    await asyncio.sleep(0.05)

                if lock_token is None and lock_available:
                    # The owner may have failed, or its lock may have expired.
                    _, lock_token = await acquire_lock(
                        cache_key,
                        ttl_seconds=settings.support_rag_response_cache_lock_ttl_seconds,
                    )
        else:
            logger.info(
                "support response cache bypass",
                extra={"event": "support_response_cache_bypass"},
            )

    try:
        if cache_key is not None and lock_token is not None and cache_knowledge_version is not None:
            cached = await _cached_support_response(
                cache_key,
                knowledge_version=cache_knowledge_version,
                request_id=request_id,
            )
            if cached is not None:
                return cached

        result = await asyncio.to_thread(
            ask,
            payload,
            user_jwt=user_jwt,
            plan=plan,
        )
        result.request_id = request_id

        if (
            cache_key is not None
            and cache_knowledge_version is not None
            and _is_cacheable_static_response(
                result,
                knowledge_version=cache_knowledge_version,
            )
        ):
            cache_value = result.model_dump(mode="json")
            cache_value["request_id"] = ""
            await set_json(
                cache_key,
                cache_value,
                ttl_seconds=settings.support_rag_response_cache_ttl_seconds,
            )

        return result
    finally:
        if cache_key is not None and lock_token is not None:
            await release_lock(cache_key, lock_token)
