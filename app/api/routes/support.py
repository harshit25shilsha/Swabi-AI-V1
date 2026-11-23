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
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.capabilities.support.conversation_store import (
    support_conversation_store,
)
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
    # Reuse the supplied conversation ID or create one for a new conversation.
    # Step 2 will use this ID to load and persist conversation history.

    # The gateway principal is the trusted source of user identity.
    # Never use payload.user_id for conversation ownership.
    trusted_user_id = getattr(_principal, "subject", None)

    if settings.ai_auth_mode == "gateway" and not trusted_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity is unavailable",
        )

    # Reuse an existing conversation ID or generate a new one.
    conversation_id = payload.conversation_id or str(uuid4())

    # New gateway requests use server-managed history.
    # Existing conversation IDs must load their history from Redis.
    if trusted_user_id and payload.conversation_id:
        conversation_history = await support_conversation_store.load_history(
            user_id=trusted_user_id,
            audience=payload.audience.value,
            conversation_id=conversation_id,
        )
    elif trusted_user_id:
        # New conversation: don't accept client-injected history.
        conversation_history = []
    else:
        # Temporary compatibility for legacy authentication mode.
        conversation_history = payload.conversation_history

    # Pass the server-loaded history into the existing support pipeline.
    support_payload = payload.model_copy(
        update={
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "user_id": None,
        }
    )

    async def finalize_response(
        response: SupportAskResponse,
    ) -> SupportAskResponse:
        """Attach the conversation ID and persist the completed exchange."""
        response.conversation_id = conversation_id

        if trusted_user_id and response.answer.strip():
            saved = await support_conversation_store.append_exchange(
                user_id=trusted_user_id,
                audience=payload.audience.value,
                conversation_id=conversation_id,
                user_message=payload.question,
                assistant_message=response.answer,
            )

            if not saved:
                logger.warning(
                    "Support response returned without conversation persistence",
                    extra={
                        "event": "support_conversation_persist_failed",
                        "request_id": request_id,
                    },
                )

        return response

    request_id = getattr(request.state, "request_id", "")
    plan: SupportAskPlan | None = None
    cache_key: str | None = None
    cache_knowledge_version: str | None = None
    lock_token: str | None = None

    if settings.enable_support_rag_response_cache:
        # Preserve static RAG caching for a genuinely new first turn.
        # Follow-up requests must be planned using their Redis history.
        is_new_conversation = not payload.conversation_id and not conversation_history

        if is_new_conversation:
            planning_payload = payload.model_copy(
                update={
                    "conversation_id": None,
                    "conversation_history": [],
                    "user_id": None,
                }
            )
        else:
            planning_payload = support_payload

        plan = await asyncio.to_thread(
            prepare_support_ask,
            planning_payload,
        )

        plan = await asyncio.to_thread(
            prepare_support_ask,
            planning_payload,
        )

        if plan.cache_eligible and plan.version_status is not None:
            cache_key = _support_cache_key(payload, plan)
            cache_knowledge_version = plan.version_status.local_version
            cached = await _cached_support_response(
                cache_key,
                knowledge_version=cache_knowledge_version,
                request_id=request_id,
            )
            if cached is not None:
                cached.conversation_id = conversation_id
                return await finalize_response(cached)

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
                        cached.conversation_id = conversation_id
                        return await finalize_response(cached)

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
                cached.conversation_id = conversation_id
                return await finalize_response(cached)

        result = await asyncio.to_thread(
            ask,
            support_payload,
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
            # Do not cache a conversation-specific ID.
            # Each caller must receive its own conversation ID.
            cache_value = result.model_dump(mode="json")
            cache_value["request_id"] = ""
            cache_value["conversation_id"] = None

            await set_json(
                cache_key,
                cache_value,
                ttl_seconds=settings.support_rag_response_cache_ttl_seconds,
            )
        result.conversation_id = conversation_id
        return await finalize_response(result)
    finally:
        if cache_key is not None and lock_token is not None:
            await release_lock(cache_key, lock_token)
