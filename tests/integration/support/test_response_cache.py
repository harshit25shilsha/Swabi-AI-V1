"""Static knowledge-base response-cache behavior for the support route."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.routes.support import _support_cache_key
from app.capabilities.support.intent_resolver import ResolvedIntent
from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.capabilities.support.prompts import SUPPORT_PROMPT_VERSION
from app.capabilities.support.schemas import (
    AnswerType,
    Citation,
    ConversationTurn,
    SupportAskRequest,
    SupportAskResponse,
)
from app.capabilities.support.service import SupportAskPlan, prepare_support_ask
from app.core.config import settings
from app.main import app
from app.rag.schemas import Audience
from app.rag.version_check import VersionStatus

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _static_plan(*, eligible: bool = True, version: str = "kb-v1") -> SupportAskPlan:
    return SupportAskPlan(
        resolved=ResolvedIntent(
            intent=LiveDataIntent.NONE,
            source="llm",
            reason="llm_none",
            use_rag=True,
            knowledge_query="cancellation policy",
        ),
        cache_eligible=eligible,
        version_status=VersionStatus(
            ok=True,
            local_version=version,
            alias_collection=f"swabi_chunks_{version}",
        ),
    )


def _static_response() -> SupportAskResponse:
    return SupportAskResponse(
        answer="The cancellation policy depends on when the trip is cancelled.",
        answer_type=AnswerType.ANSWERED,
        citations=[
            Citation(
                chunk_id="policy-1",
                section_id="cancellation",
                section_path="Policies / Cancellation",
                snippet="Cancellation terms depend on timing.",
            )
        ],
        retrieval_mode_used="hybrid",
        knowledge_version="kb-v1",
        provider="groq",
        prompt_version=SUPPORT_PROMPT_VERSION,
    )


def test_static_support_cache_key_is_stable_and_scoped():
    payload = SupportAskRequest(
        question="What is the cancellation policy?",
        audience=Audience.CUSTOMER,
    )
    plan = _static_plan()
    key = _support_cache_key(payload, plan)

    assert _support_cache_key(payload, plan) == key
    assert (
        _support_cache_key(
            payload.model_copy(update={"audience": Audience.VENDOR}),
            plan,
        )
        != key
    )
    assert _support_cache_key(payload, _static_plan(version="kb-v2")) != key

    changed_plan = _static_plan()
    changed_plan.resolved.knowledge_query = "different generated retrieval wording"
    assert _support_cache_key(payload, changed_plan) == key


def test_static_support_cache_hit_assigns_current_request_id(monkeypatch):
    monkeypatch.setattr(settings, "enable_support_rag_response_cache", True)
    monkeypatch.setattr(
        "app.api.routes.support.prepare_support_ask",
        lambda _payload: _static_plan(),
    )
    monkeypatch.setattr(
        "app.api.routes.support.get_json",
        AsyncMock(return_value=_static_response().model_dump(mode="json")),
    )

    with patch("app.api.routes.support.ask") as ask:
        response = client.post(
            "/api/v1/support/ask",
            headers={**AUTH, "X-Request-ID": "support-cache-hit"},
            json={"question": "What is the cancellation policy?", "audience": "customer"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "support-cache-hit"
    ask.assert_not_called()


def test_static_support_cache_miss_stores_success_without_request_id(monkeypatch):
    monkeypatch.setattr(settings, "enable_support_rag_response_cache", True)
    monkeypatch.setattr(
        "app.api.routes.support.prepare_support_ask",
        lambda _payload: _static_plan(),
    )
    monkeypatch.setattr(
        "app.api.routes.support.get_json",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.api.routes.support.acquire_lock",
        AsyncMock(return_value=(True, "owner-token")),
    )
    set_json = AsyncMock()
    release_lock = AsyncMock()
    monkeypatch.setattr("app.api.routes.support.set_json", set_json)
    monkeypatch.setattr("app.api.routes.support.release_lock", release_lock)

    with patch(
        "app.api.routes.support.ask",
        return_value=_static_response(),
    ) as ask:
        response = client.post(
            "/api/v1/support/ask",
            headers={**AUTH, "X-Request-ID": "support-cache-miss"},
            json={"question": "What is the cancellation policy?", "audience": "customer"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "support-cache-miss"
    ask.assert_called_once()
    set_json.assert_awaited_once()
    assert set_json.await_args.kwargs["ttl_seconds"] == (
        settings.support_rag_response_cache_ttl_seconds
    )
    assert set_json.await_args.args[1]["request_id"] == ""
    release_lock.assert_awaited_once_with(
        set_json.await_args.args[0],
        "owner-token",
    )


def test_static_support_cache_waiter_uses_owner_result(monkeypatch):
    monkeypatch.setattr(settings, "enable_support_rag_response_cache", True)
    monkeypatch.setattr(
        "app.api.routes.support.prepare_support_ask",
        lambda _payload: _static_plan(),
    )
    get_json = AsyncMock(side_effect=[None, _static_response().model_dump(mode="json")])
    monkeypatch.setattr("app.api.routes.support.get_json", get_json)
    monkeypatch.setattr(
        "app.api.routes.support.acquire_lock",
        AsyncMock(return_value=(True, None)),
    )

    with patch("app.api.routes.support.ask") as ask:
        response = client.post(
            "/api/v1/support/ask",
            headers={**AUTH, "X-Request-ID": "support-cache-waiter"},
            json={"question": "What is the cancellation policy?", "audience": "customer"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "support-cache-waiter"
    assert get_json.await_count == 2
    ask.assert_not_called()


def test_live_support_plan_bypasses_response_cache(monkeypatch):
    monkeypatch.setattr(settings, "enable_support_rag_response_cache", True)
    live_plan = SupportAskPlan(
        resolved=ResolvedIntent(
            intent=LiveDataIntent.MY_PAYMENTS,
            source="llm",
            reason="payment history",
            use_rag=False,
        ),
        cache_eligible=False,
    )
    monkeypatch.setattr(
        "app.api.routes.support.prepare_support_ask",
        lambda _payload: live_plan,
    )
    get_json = AsyncMock()
    monkeypatch.setattr("app.api.routes.support.get_json", get_json)
    live_response = SupportAskResponse(
        answer="Your latest payment is pending.",
        answer_type=AnswerType.ANSWERED,
        live_data_used=True,
        live_data_sources=["payments"],
    )

    with patch("app.api.routes.support.ask", return_value=live_response):
        response = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "Show my payments", "audience": "customer"},
        )

    assert response.status_code == 200
    get_json.assert_not_awaited()


def test_conversation_context_is_not_cache_eligible():
    resolver = MagicMock()
    resolver.resolve.return_value = _static_plan().resolved
    request = SupportAskRequest(
        question="What is the cancellation policy?",
        audience=Audience.CUSTOMER,
        conversation_history=[ConversationTurn(role="user", content="Earlier question")],
    )

    with (
        patch("app.capabilities.support.service._get_resolver", return_value=resolver),
        patch("app.capabilities.support.service._get_retriever") as get_retriever,
    ):
        plan = prepare_support_ask(request)

    assert not plan.cache_eligible
    get_retriever.assert_not_called()


def test_response_with_live_data_is_never_written_to_static_cache(monkeypatch):
    monkeypatch.setattr(settings, "enable_support_rag_response_cache", True)
    monkeypatch.setattr(
        "app.api.routes.support.prepare_support_ask",
        lambda _payload: _static_plan(),
    )
    monkeypatch.setattr(
        "app.api.routes.support.get_json",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.api.routes.support.acquire_lock",
        AsyncMock(return_value=(True, "owner-token")),
    )
    set_json = AsyncMock()
    monkeypatch.setattr("app.api.routes.support.set_json", set_json)
    mixed_response = _static_response().model_copy(
        update={"live_data_used": True, "live_data_sources": ["payments"]}
    )

    with patch("app.api.routes.support.ask", return_value=mixed_response):
        response = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "What is the cancellation policy?", "audience": "customer"},
        )

    assert response.status_code == 200
    set_json.assert_not_awaited()
