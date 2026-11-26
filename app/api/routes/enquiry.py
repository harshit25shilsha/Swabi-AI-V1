"""Enquiry Understanding route.

Thin: authenticate, invoke the service, serialize. Sync route — FastAPI
runs it in a thread pool, matching the sync service chain.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from app.capabilities.enquiry.prompts import ENQUIRY_PROMPT_VERSION
from app.capabilities.enquiry.schemas import (
    EnquiryExtractRequest,
    EnquiryUnderstandResponse,
)
from app.capabilities.enquiry.service import understand
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

router = APIRouter(prefix="/api/v1/enquiry", tags=["enquiry"])
logger = logging.getLogger(__name__)

_CACHE_NAMESPACE = "enquiry-understand:v1"


def _cache_key(
    message: str,
    currency_hint: str | None,
    previous_extraction: object | None = None,
) -> str:
    """Include every input/version that can affect an enquiry response."""
    return make_cache_key(
        _CACHE_NAMESPACE,
        {
            "message": normalize_message(message),
            "currency_hint": currency_hint.strip().upper() if currency_hint else None,
            "previous_extraction": (
                previous_extraction.model_dump(mode="json")
                if hasattr(previous_extraction, "model_dump")
                else previous_extraction
            ),
            "model": settings.llm_model,
            "prompt_version": ENQUIRY_PROMPT_VERSION,
            "response_schema_version": "v1",
        },
    )


async def _cached_result(
    key: str,
    *,
    request_id: str,
) -> EnquiryUnderstandResponse | None:
    """Deserialize a valid cached response and apply this request's ID."""
    cached = await get_json(key)
    if cached is None:
        return None

    try:
        result = EnquiryUnderstandResponse.model_validate(cached)
    except ValueError:
        logger.warning(
            "Invalid enquiry response cache payload; treating as a miss",
            extra={"event": "enquiry_response_cache_invalid"},
        )
        return None

    result.request_id = request_id
    logger.info(
        "enquiry response cache hit",
        extra={"event": "enquiry_response_cache_hit"},
    )
    return result


@router.post(
    "/understand",
    response_model=EnquiryUnderstandResponse,
    dependencies=[require_user_capability("ai:enquiry:understand")],
)
async def understand_enquiry(
    payload: EnquiryExtractRequest,
    request: Request,
) -> EnquiryUnderstandResponse:
    key: str | None = None
    lock_token: str | None = None
    request_id = getattr(request.state, "request_id", "")
    if settings.enable_enquiry_response_cache:
        key = _cache_key(
            payload.message,
            payload.currency_hint,
            payload.previous_extraction,
        )
        cached = await _cached_result(key, request_id=request_id)
        if cached is not None:
            return cached

        logger.info(
            "enquiry response cache miss",
            extra={"event": "enquiry_response_cache_miss"},
        )

        redis_available, lock_token = await acquire_lock(
            key,
            ttl_seconds=settings.enquiry_response_cache_lock_ttl_seconds,
        )
        if redis_available and lock_token is None:
            logger.info(
                "enquiry response cache lock busy; waiting for result",
                extra={"event": "enquiry_response_cache_lock_wait"},
            )
            deadline = (
                asyncio.get_running_loop().time()
                + settings.enquiry_response_cache_lock_wait_seconds
            )
            while asyncio.get_running_loop().time() < deadline:
                cached = await _cached_result(key, request_id=request_id)
                if cached is not None:
                    return cached

                _redis_available, replacement_token = await acquire_lock(
                    key,
                    ttl_seconds=settings.enquiry_response_cache_lock_ttl_seconds,
                )
                if replacement_token is not None:
                    lock_token = replacement_token
                    break
                await asyncio.sleep(0.05)

            if lock_token is None:
                # The owner may have failed, or the lock may have expired.
                # Retry once so this request can take over before failing open.
                _redis_available, lock_token = await acquire_lock(
                    key,
                    ttl_seconds=settings.enquiry_response_cache_lock_ttl_seconds,
                )

    try:
        # Recheck after owning the lock: another request might have completed
        # between this request's first cache read and lock acquisition.
        if key is not None and lock_token is not None:
            cached = await _cached_result(key, request_id=request_id)
            if cached is not None:
                return cached

        # ``understand`` calls a synchronous LLM client. Keep it off the event
        # loop now that this route awaits Redis operations.
        result = await asyncio.to_thread(
            understand,
            payload.message,
            currency_hint=payload.currency_hint,
            previous_extraction=payload.previous_extraction,
        )

        # Correlate with the outer request-id middleware.
        result.request_id = request_id

        if key is not None:
            # Never persist the request ID: each cache hit receives its own ID.
            cache_value = result.model_dump(mode="json")
            cache_value["request_id"] = ""
            await set_json(
                key,
                cache_value,
                ttl_seconds=settings.enquiry_response_cache_ttl_seconds,
            )

        # No `request.state.enquiry_result` stash: the observability
        # middleware only handles moderation results. The service already
        # emits its own `enquiry_understood` log line with request_id in
        # context, so the middleware has nothing to add here.

        return result
    finally:
        if key is not None and lock_token is not None:
            await release_lock(key, lock_token)
