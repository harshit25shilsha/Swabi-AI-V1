import httpx
import pytest
from pytest import MonkeyPatch

from app.core.auth_context import set_user_id, set_user_jwt
from app.core.exceptions import SwabiAuthenticationError
from app.swabi.client import SwabiClient


@pytest.fixture(autouse=True)
def _reset_auth_context():
    """Ensure each test starts with a clean auth context."""
    set_user_jwt(None)
    set_user_id(None)
    yield
    set_user_jwt(None)
    set_user_id(None)


@pytest.mark.asyncio
async def test_swabi_client_raises_on_401(monkeypatch: MonkeyPatch):
    client = SwabiClient(base_url="http://test", api_key="x", max_retries=0)

    async def fake_request(*args, **kwargs):
        return httpx.Response(401, json={"error": "no"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    with pytest.raises(SwabiAuthenticationError):
        await client.get_customer_profile("u1")


@pytest.mark.asyncio
async def test_swabi_client_returns_json(monkeypatch: MonkeyPatch):
    client = SwabiClient(base_url="http://test", api_key="x", max_retries=0)

    async def fake_request(*args, **kwargs):
        return httpx.Response(200, json={"user_id": "u1"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    data = await client.get_customer_profile("u1")
    assert data["user_id"] == "u1"


# --- Auth precedence tests -------------------------------------------------


def test_user_jwt_takes_precedence_over_service_key():
    client = SwabiClient(base_url="http://test", api_key="service-key")
    set_user_jwt("user-token")

    headers = client._headers()

    assert headers["Authorization"] == "Bearer user-token"


def test_service_key_used_when_no_user_jwt():
    client = SwabiClient(base_url="http://test", api_key="service-key")
    set_user_jwt(None)

    headers = client._headers()

    assert headers["Authorization"] == "Bearer service-key"


def test_no_authorization_when_neither_present():
    client = SwabiClient(base_url="http://test", api_key="")
    set_user_jwt(None)

    headers = client._headers()

    assert "Authorization" not in headers


def test_request_id_is_propagated():
    from app.core.logging import request_id_var

    client = SwabiClient(base_url="http://test", api_key="service-key")
    token = request_id_var.set("rid-123")
    try:
        headers = client._headers()
    finally:
        request_id_var.reset(token)

    assert headers["X-Request-ID"] == "rid-123"
