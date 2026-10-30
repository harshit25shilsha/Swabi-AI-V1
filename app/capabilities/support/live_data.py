"""Live Swabi backend calls for support questions.

Each fetcher:
  * calls one endpoint via the Swabi backend
  * forwards the user's JWT as Bearer (public intents may pass None)
  * unwraps the ``{"success", "data", ...}`` envelope
  * decrypts the envelope first if the backend returns ``{"payload": "..."}``
  * returns the payload the formatter consumes
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.core.config import settings
from app.core.logging import get_logger, request_id_var
from app.swabi.crypto import SwabiDecryptionError, decrypt_payload

log = get_logger(__name__)

LIVE_FETCH_TIMEOUT = 10  # seconds
LIVE_FETCH_MAX_ATTEMPTS = 2
LIVE_FETCH_RETRY_DELAY_SECONDS = 0.25

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_PACKAGE_CODE_RE = re.compile(r"^PKG-\d+$", re.IGNORECASE)
_PACKAGE_LOOKUP_PAGE_SIZE = 100
_PACKAGE_LOOKUP_MAX_PAGES = 10


class LiveDataFetchError(Exception):
    pass


# Public intents — no Authorization header required.
_PUBLIC_INTENTS: set[LiveDataIntent] = {
    LiveDataIntent.PACKAGE_SEARCH,
    LiveDataIntent.PACKAGE_DETAIL,
    LiveDataIntent.ACTIVITY_SEARCH,
    LiveDataIntent.POPULAR_DESTINATIONS,
}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _headers(user_jwt: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Request-ID": request_id_var.get(),
    }
    if user_jwt:
        headers["Authorization"] = f"Bearer {user_jwt}"
    return headers


def _get(
    path: str,
    *,
    user_jwt: str | None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{settings.swabi_api_base_url.rstrip('/')}{path}"
    for attempt in range(1, LIVE_FETCH_MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=LIVE_FETCH_TIMEOUT) as client:
                resp = client.get(
                    url, headers=_headers(user_jwt), params=params or {}
                )
            break
        except httpx.TimeoutException as exc:
            if attempt == LIVE_FETCH_MAX_ATTEMPTS:
                raise LiveDataFetchError(f"{path}: timeout") from exc
            log.warning(
                "live_data_timeout_retry",
                extra={
                    "path": path,
                    "attempt": attempt,
                    "max_attempts": LIVE_FETCH_MAX_ATTEMPTS,
                },
            )
            time.sleep(LIVE_FETCH_RETRY_DELAY_SECONDS)
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

    if set(body.keys()) == {"payload"}:
        if not settings.swabi_response_passphrase:
            raise LiveDataFetchError(
                f"{path}: encrypted response but "
                "SWABI_RESPONSE_PASSPHRASE is not configured"
            )
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


# ---------------------------------------------------------------------------
# Customer fetchers
# ---------------------------------------------------------------------------

def fetch_my_bookings(
    user_jwt: str, status: str = "all"
) -> dict[str, Any]:
    response = _get(
        "/bookings/my", user_jwt=user_jwt, params={"status": status}
    )
    data = _unwrap(response) or {}
    return {
        "bookings": data.get("bookings", []),
        "total": data.get("total", 0),
    }


def fetch_my_booking_detail(
    user_jwt: str, booking_id: str
) -> dict[str, Any]:
    response = _get(f"/bookings/my/{booking_id}", user_jwt=user_jwt)
    return {"booking": _unwrap(response) or {}}


def fetch_my_payments(
    user_jwt: str, status: str | None = None
) -> dict[str, Any]:
    params = {"status": status} if status else {}
    response = _get(
        "/payments/my-payments",
        user_jwt=user_jwt,
        params=params,
    )
    data = _unwrap(response) or []
    if isinstance(data, dict):
        data = data.get("payments", [])
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


def fetch_enquiry_bids(
    user_jwt: str, enquiry_id: str
) -> dict[str, Any]:
    response = _get(
        f"/bids/enquiry/{enquiry_id}",
        user_jwt=user_jwt,
        params={"limit": 50, "offset": 0},
    )
    data = _unwrap(response) or {}
    return {
        "bids": data.get("bids", []),
        "total": data.get("total", 0),
    }


# ---------------------------------------------------------------------------
# Public catalogue fetchers (no auth required)
# ---------------------------------------------------------------------------

def fetch_package_list(
    user_jwt: str | None,
    destination: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"page": 1, "limit": 10}
    if destination:
        params["destination"] = destination
    response = _get("/packages", user_jwt=user_jwt, params=params)
    data = _unwrap(response) or {}
    return {
        "packages": data.get("packages", []),
        "total": data.get("total", 0),
    }


def fetch_package_detail(
    user_jwt: str | None, package_id: str
) -> dict[str, Any]:
    response = _get(f"/packages/{package_id}", user_jwt=user_jwt)
    return {"package": _unwrap(response) or {}}


def fetch_package_by_code(
    user_jwt: str | None, package_code: str
) -> dict[str, Any]:
    """Resolve the public package code through the paginated catalogue."""
    if not _PACKAGE_CODE_RE.fullmatch(package_code):
        raise LiveDataFetchError("invalid package code")

    examined = 0
    total = 0
    for page in range(1, _PACKAGE_LOOKUP_MAX_PAGES + 1):
        response = _get(
            "/packages",
            user_jwt=user_jwt,
            params={"page": page, "limit": _PACKAGE_LOOKUP_PAGE_SIZE},
        )
        data = _unwrap(response)
        if not isinstance(data, dict):
            raise LiveDataFetchError(
                "/packages: unexpected package catalogue response"
            )
        packages = data.get("packages")
        if not isinstance(packages, list):
            raise LiveDataFetchError(
                "/packages: package catalogue is missing its packages list"
            )
        try:
            total = int(data.get("total", len(packages)))
        except (TypeError, ValueError) as exc:
            raise LiveDataFetchError(
                "/packages: invalid package catalogue total"
            ) from exc

        for package in packages:
            if not isinstance(package, dict):
                continue
            code = package.get("packageCode")
            if isinstance(code, str) and code.casefold() == package_code.casefold():
                return {"package": package}

        examined += len(packages)
        if not packages or examined >= total:
            return {"package": {}, "not_found_code": package_code}

    if examined < total:
        raise LiveDataFetchError(
            "/packages: package code lookup exceeded the safe page limit"
        )
    return {"package": {}, "not_found_code": package_code}


def fetch_activity_list(
    user_jwt: str | None,
    query: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Public activities. Endpoint changed from /activities to
    /activities/public/list — do not revert."""
    params: dict[str, Any] = {"page": 1, "limit": 10}
    if query:
        params["search"] = query
    if category:
        params["category"] = category
    response = _get(
        "/activities/public/list", user_jwt=user_jwt, params=params
    )
    data = _unwrap(response) or {}
    return {
        "activities": data.get("activities", []),
        "total": data.get("total", 0),
    }


def fetch_popular_destinations(
    user_jwt: str | None,
) -> dict[str, Any]:
    response = _get("/analytics/popular-destinations", user_jwt=user_jwt)
    data = _unwrap(response) or []
    return {"destinations": data if isinstance(data, list) else []}


# ---------------------------------------------------------------------------
# Vendor fetchers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FETCHERS: dict[LiveDataIntent, Callable[..., Any]] = {
    LiveDataIntent.MY_BOOKINGS:          fetch_my_bookings,
    LiveDataIntent.MY_BOOKING_DETAIL:    fetch_my_booking_detail,
    LiveDataIntent.MY_PAYMENTS:          fetch_my_payments,
    LiveDataIntent.MY_COINS:             fetch_my_coins,
    LiveDataIntent.MY_COIN_HISTORY:      fetch_my_coin_history,
    LiveDataIntent.MY_ENQUIRIES:         fetch_my_enquiries,
    LiveDataIntent.MY_ENQUIRY_BIDS:      fetch_enquiry_bids,
    LiveDataIntent.PACKAGE_SEARCH:       fetch_package_list,
    LiveDataIntent.PACKAGE_DETAIL:       fetch_package_detail,
    LiveDataIntent.ACTIVITY_SEARCH:      fetch_activity_list,
    LiveDataIntent.POPULAR_DESTINATIONS: fetch_popular_destinations,
    LiveDataIntent.VENDOR_WALLET:        fetch_vendor_wallet,
    LiveDataIntent.VENDOR_TRANSACTIONS:  fetch_vendor_transactions,
    LiveDataIntent.VENDOR_WITHDRAWALS:   fetch_vendor_withdrawals,
    LiveDataIntent.VENDOR_INQUIRIES:     fetch_vendor_inquiries,
    LiveDataIntent.VENDOR_BOOKINGS:      fetch_vendor_bookings,
    LiveDataIntent.VENDOR_BIDS:          fetch_vendor_bids,
    LiveDataIntent.VENDOR_SUBSCRIPTION:  fetch_vendor_subscription,
}

_ENDPOINT_LABEL: dict[LiveDataIntent, str] = {
    LiveDataIntent.MY_BOOKINGS:          "GET /api/bookings/my",
    LiveDataIntent.MY_BOOKING_DETAIL:    "GET /api/bookings/my/{id}",
    LiveDataIntent.MY_PAYMENTS:          "GET /api/payments/my-payments",
    LiveDataIntent.MY_COINS:             "GET /api/wallet/me/balance",
    LiveDataIntent.MY_COIN_HISTORY:      "GET /api/wallet/me/transactions",
    LiveDataIntent.MY_ENQUIRIES:         "GET /api/enquiries/my",
    LiveDataIntent.MY_ENQUIRY_BIDS:      "GET /api/bids/enquiry/{enquiryId}",
    LiveDataIntent.PACKAGE_SEARCH:       "GET /api/packages",
    LiveDataIntent.PACKAGE_DETAIL:       "GET /api/packages/{id}",
    LiveDataIntent.ACTIVITY_SEARCH:      "GET /api/activities/public/list",
    LiveDataIntent.POPULAR_DESTINATIONS: "GET /api/analytics/popular-destinations",
    LiveDataIntent.VENDOR_WALLET:        "GET /api/vendor/wallet/balance",
    LiveDataIntent.VENDOR_TRANSACTIONS:  "GET /api/vendor/wallet/transactions",
    LiveDataIntent.VENDOR_WITHDRAWALS:   "GET /api/vendor/wallet/withdrawals",
    LiveDataIntent.VENDOR_INQUIRIES:     "GET /api/vendor-inquiries",
    LiveDataIntent.VENDOR_BOOKINGS:      "GET /api/bookings",
    LiveDataIntent.VENDOR_BIDS:          "GET /api/bids/vendor/my-bids",
    LiveDataIntent.VENDOR_SUBSCRIPTION:  "GET /api/subscriptions/my",
}

_ID_REQUIRED: dict[LiveDataIntent, str] = {
    LiveDataIntent.MY_BOOKING_DETAIL: "booking_id",
    LiveDataIntent.MY_ENQUIRY_BIDS:   "enquiry_id",
    LiveDataIntent.PACKAGE_DETAIL:    "package_id",
}

_ID_FALLBACK: dict[LiveDataIntent, LiveDataIntent] = {
    LiveDataIntent.MY_BOOKING_DETAIL: LiveDataIntent.MY_BOOKINGS,
    LiveDataIntent.PACKAGE_DETAIL:    LiveDataIntent.PACKAGE_SEARCH,
}

_STATUS_DEFAULTS: dict[LiveDataIntent, str] = {
    LiveDataIntent.MY_BOOKINGS: "all",
}

_STATUS_ALIASES = {
    "cancel": "cancelled",
    "canceled": "cancelled",
    "cancelled": "cancelled",
    "complete": "completed",
    "completed": "completed",
}


def _created_at_timestamp(enquiry: dict[str, Any]) -> float:
    created_at = enquiry.get("createdAt")
    if not isinstance(created_at, str) or not created_at.strip():
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def fetch_for_intent(
    intent: LiveDataIntent,
    *,
    user_jwt: str | None,
    entities: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Dispatch to the right fetcher. Returns ``(endpoint_label, payload)``."""
    if intent not in _FETCHERS:
        raise LiveDataFetchError(f"no fetcher for intent {intent.value}")
    if not user_jwt and intent not in _PUBLIC_INTENTS:
        raise LiveDataFetchError("user JWT required")

    entities = entities or {}
    handler = _FETCHERS[intent]

    if intent == LiveDataIntent.MY_ENQUIRY_BIDS:
        enquiry_id = entities.get("enquiry_id") or entities.get("id")
        enquiry_id_str = str(enquiry_id) if enquiry_id is not None else None
        if enquiry_id_str and _UUID_RE.match(enquiry_id_str):
            payload = handler(user_jwt, enquiry_id_str)
            return _ENDPOINT_LABEL[intent], payload

        enquiries_payload = fetch_my_enquiries(user_jwt)
        enquiries = [
            item for item in enquiries_payload["enquiries"]
            if isinstance(item, dict) and not item.get("isDeleted", False)
        ]
        if not enquiries:
            return (
                "GET /api/enquiries/my",
                {"enquiry": None, "bids": [], "total": 0},
            )

        latest = max(enquiries, key=_created_at_timestamp)
        latest_id = latest.get("id")
        if not isinstance(latest_id, str) or not _UUID_RE.match(latest_id):
            raise LiveDataFetchError(
                "most recent enquiry does not contain a valid ID"
            )

        bids = handler(user_jwt, latest_id)
        return (
            "GET /api/enquiries/my + GET /api/bids/enquiry/{enquiryId}",
            {
                "enquiry": {
                    "id": latest_id,
                    "title": latest.get("title"),
                    "destination": latest.get("destination"),
                    "createdAt": latest.get("createdAt"),
                },
                **bids,
            },
        )

    # ID-required intents (detail endpoints).
    if intent in _ID_REQUIRED:
        key = _ID_REQUIRED[intent]
        entity_id = entities.get(key) or entities.get("id")
        entity_id_str = str(entity_id) if entity_id is not None else None
        package_code = entities.get("package_code")
        if (
            intent == LiveDataIntent.PACKAGE_DETAIL
            and not package_code
            and entity_id_str
            and _PACKAGE_CODE_RE.fullmatch(entity_id_str)
        ):
            package_code = entity_id_str
        if (
            intent == LiveDataIntent.PACKAGE_DETAIL
            and package_code
        ):
            payload = fetch_package_by_code(
                user_jwt, str(package_code)
            )
            return "GET /api/packages (lookup by package code)", payload
        if entity_id_str and _UUID_RE.match(entity_id_str):
            payload = handler(user_jwt, entity_id_str)
            return _ENDPOINT_LABEL[intent], payload

        fallback = _ID_FALLBACK[intent]
        log.info(
            "live_data_id_fallback",
            extra={
                "requested": intent.value,
                "fallback": fallback.value,
                "entity_id": entity_id_str,
            },
        )
        payload = _FETCHERS[fallback](user_jwt)
        return _ENDPOINT_LABEL[fallback], payload

    # Public catalogue intents with optional filters.
    if intent == LiveDataIntent.PACKAGE_SEARCH:
        destination = entities.get("destination") or entities.get("query")
        payload = handler(
            user_jwt, destination=destination or None
        )
        return _ENDPOINT_LABEL[intent], payload

    if intent == LiveDataIntent.ACTIVITY_SEARCH:
        query = entities.get("query") or None
        category = entities.get("category") or None
        payload = handler(user_jwt, query=query, category=category)
        return _ENDPOINT_LABEL[intent], payload

    if intent == LiveDataIntent.POPULAR_DESTINATIONS:
        payload = handler(user_jwt)
        return _ENDPOINT_LABEL[intent], payload

    # Status-aware intents.
    if intent == LiveDataIntent.MY_PAYMENTS:
        status = entities.get("status")
        if status:
            status = _STATUS_ALIASES.get(str(status).strip().lower(), status)
            payload = handler(user_jwt, status=status)
        else:
            payload = handler(user_jwt)
        return _ENDPOINT_LABEL[intent], payload

    if intent in _STATUS_DEFAULTS:
        status = entities.get("status") or _STATUS_DEFAULTS[intent]
        status = _STATUS_ALIASES.get(str(status).strip().lower(), status)
        payload = handler(user_jwt, status=status)
        return _ENDPOINT_LABEL[intent], payload

    # Everything else: no extra arguments.
    payload = handler(user_jwt)
    return _ENDPOINT_LABEL[intent], payload
