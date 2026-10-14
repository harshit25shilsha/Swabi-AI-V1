"""Enquiry Understanding route.

Thin: authenticate, invoke the service, serialize. Sync route — FastAPI
runs it in a thread pool, matching the sync service chain.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.capabilities.enquiry.schemas import (
    EnquiryExtractRequest,
    EnquiryUnderstandResponse,
)
from app.capabilities.enquiry.service import understand
from app.core.service_auth import verify_service_token

router = APIRouter(prefix="/api/v1/enquiry", tags=["enquiry"])


@router.post(
    "/understand",
    response_model=EnquiryUnderstandResponse,
    dependencies=[Depends(verify_service_token)],
)
def understand_enquiry(
    payload: EnquiryExtractRequest,
    request: Request,
) -> EnquiryUnderstandResponse:
    result = understand(
        payload.message,
        currency_hint=payload.currency_hint,
    )

    # Correlate with the outer request-id middleware.
    result.request_id = getattr(request.state, "request_id", "")

    # No `request.state.enquiry_result` stash: the observability
    # middleware only handles moderation results. The service already
    # emits its own `enquiry_understood` log line with request_id in
    # context, so the middleware has nothing to add here.

    return result
