"""Chat Safety route. Thin: authenticate, invoke service, serialize."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, Request

from app.capabilities.chat_safety.dependencies import verify_token
from app.capabilities.chat_safety.prompts import PROMPT_VERSION
from app.capabilities.chat_safety.schemas import (
    ModerationRequest,
    ModerationResponse,
)
from app.capabilities.chat_safety.service import moderate

router = APIRouter(prefix="/api/v1", tags=["chat_safety"])


@router.post(
    "/validate-message",
    response_model=ModerationResponse,
    dependencies=[Depends(verify_token)],
)
def validate_message(payload: ModerationRequest, request: Request) -> ModerationResponse:
    # Sync route: FastAPI runs this in a thread pool. `moderate` is sync.
    result = moderate(payload.message)

    # Attach correlation fields. The request_id comes from the outer
    # request-id middleware via request.state.
    result.request_id = getattr(request.state, "request_id", "")
    result.prompt_version = PROMPT_VERSION

    # Stash for the observability middleware to log & count.
    request.state.moderation_result = result
    request.state.message_hash = hashlib.sha256(payload.message.encode("utf-8")).hexdigest()[:16]

    return result
