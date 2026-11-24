"""Shared read-only transport for user-delegated Swabi (NestJS) GET calls.

Forwards the opaque user JWT, retries transient failures, unwraps the
{"success","data"} envelope and decrypts {"payload": ...} responses.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import SwabiAIError
from app.core.logging import get_logger, request_id_var
from app.swabi.crypto import SwabiDecryptionError, decrypt_payload

log = get_logger(__name__)

READ_TIMEOUT = 8
READ_MAX_ATTEMPTS = 3
READ_RETRY_DELAY = 1.0
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

_client: httpx.Client | None = None
_lock = threading.Lock()


class SwabiReadError(SwabiAIError):
    """A delegated Swabi read failed."""


class SwabiNotFoundError(SwabiReadError):
    """The requested resource does not exist or is not visible to the user."""


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = httpx.Client(
                    timeout=httpx.Timeout(READ_TIMEOUT),
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=30.0,
                    ),
                )
    return _client


def close_read_client() -> None:
    global _client
    with _lock:
        client, _client = _client, None
    if client is not None:
        client.close()


def get_json(
    path: str,
    *,
    user_jwt: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET ``path`` as the user and return the decoded body (dict)."""
    if not user_jwt:
        raise SwabiReadError("user JWT required")

    url = f"{settings.swabi_api_base_url.rstrip('/')}{path}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {user_jwt}",
        "X-Request-ID": request_id_var.get(),
    }
    client = _get_client()

    resp: httpx.Response | None = None
    for attempt in range(1, READ_MAX_ATTEMPTS + 1):
        try:
            resp = client.get(url, headers=headers, params=params or {})
        except httpx.TimeoutException as exc:
            if attempt == READ_MAX_ATTEMPTS:
                raise SwabiReadError(f"{path}: timeout") from exc
            time.sleep(READ_RETRY_DELAY * 2 ** (attempt - 1))
            continue
        except httpx.HTTPError as exc:
            raise SwabiReadError(f"{path}: {type(exc).__name__}") from exc

        if resp.status_code in _RETRYABLE and attempt < READ_MAX_ATTEMPTS:
            time.sleep(READ_RETRY_DELAY * 2 ** (attempt - 1))
            continue
        break

    assert resp is not None
    if resp.status_code in (401, 403):
        raise SwabiReadError(f"{path}: HTTP {resp.status_code} not authorized")
    if resp.status_code == 404:
        raise SwabiNotFoundError(f"{path}: not found")
    if resp.status_code >= 400:
        raise SwabiReadError(f"{path}: HTTP {resp.status_code}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise SwabiReadError(f"{path}: non-JSON response") from exc
    if not isinstance(body, dict):
        raise SwabiReadError(f"{path}: unexpected response type")

    if set(body.keys()) == {"payload"}:
        if not settings.swabi_response_passphrase:
            raise SwabiReadError(f"{path}: encrypted response, passphrase not configured")
        try:
            body = json.loads(
                decrypt_payload(
                    body["payload"],
                    settings.swabi_response_passphrase,
                    kdf=settings.swabi_response_kdf,
                )
            )
        except (SwabiDecryptionError, json.JSONDecodeError) as exc:
            raise SwabiReadError(f"{path}: could not decode encrypted response") from exc
        if not isinstance(body, dict):
            raise SwabiReadError(f"{path}: decrypted response is not an object")

    if not body.get("success", True):
        raise SwabiReadError(f"{path}: {body.get('message', 'error')}")
    return body