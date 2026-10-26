"""Live Swabi backend calls for support questions.

Each fetcher:
  * calls one endpoint via the Swabi backend
  * forwards the user's JWT as Bearer
  * unwraps the `{"success", "data", ...}` envelope
  * decrypts the envelope first if the backend returns `{"payload": "..."}`
  * returns the `data` payload unchanged

Uses ``httpx.Client`` directly (sync) instead of ``SwabiClient`` because
``SwabiClient._request`` is async while the support service is sync. The
header construction here mirrors ``SwabiClient._headers()`` so the two
clients stay behaviourally compatible.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.core.config import settings
from app.core.logging import get_logger, request_id_var
from app.swabi.crypto import SwabiDecryptionError, decrypt_payload

log = get_logger(__name__)

LIVE_FETCH_TIMEOUT = 10  # seconds


class LiveDataFetchError(Exception):
    pass


# Transport

def _headers(user_jwt: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Request-ID": request_id_var.get(),
        "Authorization": f"Bearer {user_jwt}",
    }


def _get(
    path: str,
    *,
    user_jwt: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET through the Swabi backend, forwarding the user's JWT as Bearer.

    Handles both a plain JSON envelope and an encrypted
    ``{"payload": "<base64-ciphertext>"}`` response. The passphrase is
    read from ``settings.swabi_response_passphrase``; if it is empty,
    encrypted responses are treated as unexpected.
    """
    url = f"{settings.swabi_api_base_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=LIVE_FETCH_TIMEOUT) as client:
            resp = client.get(
                url, headers=_headers(user_jwt), params=params or {}
            )
    except httpx.TimeoutException as exc:
        raise LiveDataFetchError(f"{path}: timeout") from exc
    except httpx.HTTPError as exc:
        raise LiveDataFetchError(f"{path}: {exc}") from exc

    if resp.status_code == 401:
        raise LiveDataFetchError(f"{path}: 401 unauthorized")
    if resp.status_code >= 400:
        raise LiveDataFetchError(
            f"{path}: HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        body = resp.json()
    except ValueError as exc:
        raise LiveDataFetchError(f"{path}: non-JSON response") from exc

    if not isinstance(body, dict):
        raise LiveDataFetchError(f"{path}: unexpected response type")

    # Decrypt if the backend returns an encrypted envelope.
    if (
        set(body.keys()) == {"payload"}
        and settings.swabi_response_passphrase
    ):
        log.info(
            "live_data_decrypt",
            extra={
                "path": path,
                "ciphertext_len": len(body["payload"]),
            },
        )
        try:
            plaintext = decrypt_payload(
                body["payload"],
                settings.swabi_response_passphrase,
                kdf=settings.swabi_response_kdf,
            )
        except SwabiDecryptionError as exc:
            raise LiveDataFetchError(
                f"{path}: decryption failed: {exc}"
            ) from exc
        try:
            body = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise LiveDataFetchError(
                f"{path}: decrypted payload is not JSON: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise LiveDataFetchError(
                f"{path}: decrypted response is not a dict"
            )

    if not body.get("success", True):
        raise LiveDataFetchError(
            f"{path}: {body.get('message', 'error')}"
        )

    return body


def _unwrap(response: dict[str, Any]) -> Any:
    return response.get("data")


# Customer fetchers

def fetch_my_bookings(user_jwt: str, status: str = "all") -> dict[str, Any]:
    response = _get(
        "/bookings/my", user_jwt=user_jwt, params={"status": status}
    )
    data = _unwrap(response) or {}
    return {
        "bookings": data.get("bookings", []),
        "total": data.get("total", 0),
    }


def fetch_my_payments(
    user_jwt: str, status: str = "completed"
) -> dict[str, Any]:
    response = _get(
        "/payments/my-payments",
        user_jwt=user_jwt,
        params={"status": status},
    )
    data = _unwrap(response) or []
    return {
        "payments": data if isinstance(data, list) else [],
        "total": len(data or []),
    }


def fetch_my_coins(user_jwt: str) -> dict[str, Any]:
    response = _get("/wallet/me/balance", user_jwt=user_jwt)
    return _unwrap(response) or {}


def fetch_my_coin_history(user_jwt: str) -> dict[str, Any]:
    response = _get("/wallet/me/transactions", user_jwt=user_jwt)
    data = _unwrap(response) or []
    meta = response.get("meta") or {}
    return {
        "transactions": data if isinstance(data, list) else [],
        "total": meta.get("total", 0),
    }


def fetch_my_enquiries(user_jwt: str) -> dict[str, Any]:
    response = _get("/enquiries/my", user_jwt=user_jwt)
    data = _unwrap(response) or []
    return {
        "enquiries": data if isinstance(data, list) else [],
        "total": len(data or []),
    }



# Vendor fetchers

def fetch_vendor_wallet(user_jwt: str) -> dict[str, Any]:
    response = _get("/vendor/wallet/balance", user_jwt=user_jwt)
    return _unwrap(response) or {}


def fetch_vendor_transactions(user_jwt: str) -> dict[str, Any]:
    response = _get("/vendor/wallet/transactions", user_jwt=user_jwt)
    data = _unwrap(response) or []
    meta = response.get("meta") or {}
    return {
        "transactions": data if isinstance(data, list) else [],
        "total": meta.get("total", 0),
    }


def fetch_vendor_withdrawals(user_jwt: str) -> dict[str, Any]:
    response = _get("/vendor/wallet/withdrawals", user_jwt=user_jwt)
    data = _unwrap(response) or []
    meta = response.get("meta") or {}
    return {
        "withdrawals": data if isinstance(data, list) else [],
        "total": meta.get("total", 0),
    }


def fetch_vendor_inquiries(user_jwt: str) -> dict[str, Any]:
    response = _get("/vendor-inquiries", user_jwt=user_jwt)
    data = _unwrap(response) or {}
    return {
        "inquiries": data.get("inquiries", []),
        "total": data.get("total", 0),
        "stats": data.get("stats", {}),
    }


def fetch_vendor_bookings(user_jwt: str) -> dict[str, Any]:
    response = _get("/bookings", user_jwt=user_jwt)
    data = _unwrap(response) or {}
    return {
        "bookings": data.get("bookings", []),
        "total": data.get("total", 0),
    }


def fetch_vendor_bids(user_jwt: str) -> dict[str, Any]:
    response = _get("/bids/vendor/my-bids", user_jwt=user_jwt)
    data = _unwrap(response) or {}
    return {
        "bids": data.get("bids", []),
        "total": data.get("total", 0),
        "stats": data.get("stats", {}),
    }


def fetch_vendor_subscription(user_jwt: str) -> Any:
    response = _get("/subscriptions/my", user_jwt=user_jwt)
    return _unwrap(response)


# Dispatch

_FETCHERS = {
    LiveDataIntent.MY_BOOKINGS:         fetch_my_bookings,
    LiveDataIntent.MY_PAYMENTS:         fetch_my_payments,
    LiveDataIntent.MY_COINS:            fetch_my_coins,
    LiveDataIntent.MY_COIN_HISTORY:     fetch_my_coin_history,
    LiveDataIntent.MY_ENQUIRIES:        fetch_my_enquiries,
    LiveDataIntent.VENDOR_WALLET:       fetch_vendor_wallet,
    LiveDataIntent.VENDOR_TRANSACTIONS: fetch_vendor_transactions,
    LiveDataIntent.VENDOR_WITHDRAWALS:  fetch_vendor_withdrawals,
    LiveDataIntent.VENDOR_INQUIRIES:    fetch_vendor_inquiries,
    LiveDataIntent.VENDOR_BOOKINGS:     fetch_vendor_bookings,
    LiveDataIntent.VENDOR_BIDS:         fetch_vendor_bids,
    LiveDataIntent.VENDOR_SUBSCRIPTION: fetch_vendor_subscription,
}

_ENDPOINT_LABEL = {
    LiveDataIntent.MY_BOOKINGS:         "GET /api/bookings/my",
    LiveDataIntent.MY_PAYMENTS:         "GET /api/payments/my-payments",
    LiveDataIntent.MY_COINS:            "GET /api/wallet/me/balance",
    LiveDataIntent.MY_COIN_HISTORY:     "GET /api/wallet/me/transactions",
    LiveDataIntent.MY_ENQUIRIES:        "GET /api/enquiries/my",
    LiveDataIntent.VENDOR_WALLET:       "GET /api/vendor/wallet/balance",
    LiveDataIntent.VENDOR_TRANSACTIONS: "GET /api/vendor/wallet/transactions",
    LiveDataIntent.VENDOR_WITHDRAWALS:  "GET /api/vendor/wallet/withdrawals",
    LiveDataIntent.VENDOR_INQUIRIES:    "GET /api/vendor-inquiries",
    LiveDataIntent.VENDOR_BOOKINGS:     "GET /api/bookings",
    LiveDataIntent.VENDOR_BIDS:         "GET /api/bids/vendor/my-bids",
    LiveDataIntent.VENDOR_SUBSCRIPTION: "GET /api/subscriptions/my",
}


def fetch_for_intent(
    intent: LiveDataIntent,
    *,
    user_jwt: str | None,
) -> tuple[str, dict[str, Any]]:
    """Dispatch to the right fetcher. Returns (endpoint_label, payload).

    Raises LiveDataFetchError on transport error, non-200, bad envelope,
    or decryption failure.
    """
    if intent not in _FETCHERS:
        raise LiveDataFetchError(f"no fetcher for intent {intent.value}")
    if not user_jwt:
        raise LiveDataFetchError("user JWT required")

    payload = _FETCHERS[intent](user_jwt)
    return _ENDPOINT_LABEL[intent], payload
