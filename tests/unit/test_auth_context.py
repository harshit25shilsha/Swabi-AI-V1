from app.core.auth_context import (
    get_user_id,
    get_user_jwt,
    set_user_id,
    set_user_jwt,
)
from app.main import _extract_bearer_token


def test_set_and_get_user_jwt():
    set_user_jwt("abc")
    assert get_user_jwt() == "abc"
    set_user_jwt(None)
    assert get_user_jwt() is None


def test_set_and_get_user_id():
    set_user_id("u1")
    assert get_user_id() == "u1"
    set_user_id(None)
    assert get_user_id() is None


def test_extract_bearer_valid():
    assert _extract_bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"
    assert _extract_bearer_token("bearer abc") == "abc"


def test_extract_bearer_rejects_non_bearer():
    assert _extract_bearer_token("Basic dXNlcjpwYXNz") is None
    assert _extract_bearer_token("") is None
    assert _extract_bearer_token("Bearer") is None
    assert _extract_bearer_token("Bearer   ") is None
    assert _extract_bearer_token("Token abc") is None
