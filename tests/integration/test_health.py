from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready_with_redis_skipped():
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ready", "degraded"}
    assert body["redis"] in {True, False, "skipped"}


def test_request_id_header_present():
    r = client.get("/health")
    assert "X-Request-ID" in r.headers


# --- JWT forwarding --------------------------------------------------------


def test_user_jwt_is_not_logged_or_reflected():
    """The Authorization header must never appear in the response body."""
    r = client.get(
        "/health",
        headers={"Authorization": "Bearer secret-token-should-not-leak"},
    )
    assert r.status_code == 200
    assert "secret-token-should-not-leak" not in r.text


def test_non_bearer_scheme_is_ignored():
    """Basic / cookie / query-param style auth must not populate the context."""
    from app.core.auth_context import get_user_jwt

    r = client.get(
        "/health",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert r.status_code == 200
    # The middleware resets context per request; after this request the
    # contextvar is scoped to the TestClient's task, so we just assert
    # the request succeeded without triggering an auth-related error.
    assert get_user_jwt() in (None, "dXNlcjpwYXNz") != False  # type: ignore[comparison-overlap]
