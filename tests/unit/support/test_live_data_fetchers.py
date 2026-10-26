"""Fetcher tests. httpx.Client is mocked; no network, no real backend."""
from __future__ import annotations

import httpx
import pytest

from app.capabilities.support import live_data
from app.capabilities.support.live_data import LiveDataFetchError
from app.capabilities.support.live_data_classifier import LiveDataIntent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, body, *, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text or str(body)

    def json(self):
        return self._body


def _mock_httpx(
    monkeypatch,
    *,
    status_code: int = 200,
    body=None,
    raises: Exception | None = None,
    record: list | None = None,
):
    """Replace live_data.httpx.Client with a fake returning the given body."""

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None, params=None):
            if record is not None:
                record.append(
                    {"url": url, "headers": headers, "params": params}
                )
            if raises is not None:
                raise raises
            return _FakeResponse(status_code, body or {})

    monkeypatch.setattr(
        "app.capabilities.support.live_data.httpx.Client", FakeClient
    )


def _jwt() -> str:
    return "test-user-jwt"


# ---------------------------------------------------------------------------
# Customer fetchers
# ---------------------------------------------------------------------------

def test_fetch_my_bookings(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {"bookings": [{"bookingCode": "BK-1"}], "total": 1},
        },
        record=record,
    )
    result = live_data.fetch_my_bookings(_jwt())
    assert result["total"] == 1
    assert record[0]["url"].endswith("/bookings/my")
    assert record[0]["params"]["status"] == "all"
    assert record[0]["headers"]["Authorization"] == f"Bearer {_jwt()}"


def test_fetch_my_bookings_custom_status(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": {"bookings": [], "total": 0}},
        record=record,
    )
    live_data.fetch_my_bookings(_jwt(), status="upcoming")
    assert record[0]["params"]["status"] == "upcoming"


def test_fetch_my_payments(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [{"id": "p1"}, {"id": "p2"}]},
        record=record,
    )
    result = live_data.fetch_my_payments(_jwt())
    assert result["total"] == 2
    assert record[0]["url"].endswith("/payments/my-payments")


def test_fetch_my_coins(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": {"coinBalance": 100}},
        record=record,
    )
    assert live_data.fetch_my_coins(_jwt())["coinBalance"] == 100
    assert record[0]["url"].endswith("/wallet/me/balance")


def test_fetch_my_coin_history(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": [{"id": "t1"}],
            "meta": {"total": 1},
        },
    )
    result = live_data.fetch_my_coin_history(_jwt())
    assert result["total"] == 1
    assert len(result["transactions"]) == 1


def test_fetch_my_enquiries(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [{"id": "e1"}, {"id": "e2"}]},
    )
    assert live_data.fetch_my_enquiries(_jwt())["total"] == 2


# ---------------------------------------------------------------------------
# Vendor fetchers
# ---------------------------------------------------------------------------

def test_fetch_vendor_wallet(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {"escrowBalance": 500, "topupBalance": 1000},
        },
    )
    result = live_data.fetch_vendor_wallet(_jwt())
    assert result["escrowBalance"] == 500


def test_fetch_vendor_transactions(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [], "meta": {"total": 0}},
    )
    assert live_data.fetch_vendor_transactions(_jwt())["total"] == 0


def test_fetch_vendor_withdrawals(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [], "meta": {"total": 0}},
    )
    assert live_data.fetch_vendor_withdrawals(_jwt())["total"] == 0


def test_fetch_vendor_inquiries(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {
                "inquiries": [{"id": "i1"}],
                "total": 1,
                "stats": {
                    "new": 1, "bid_placed": 0, "won": 0, "lost": 0, "all": 1,
                },
            },
        },
    )
    result = live_data.fetch_vendor_inquiries(_jwt())
    assert result["total"] == 1
    assert result["stats"]["new"] == 1


def test_fetch_vendor_bookings(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {"bookings": [{"bookingCode": "BK-2"}], "total": 1},
        },
    )
    assert live_data.fetch_vendor_bookings(_jwt())["total"] == 1


def test_fetch_vendor_bids(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {
                "bids": [{"id": "b1"}],
                "total": 1,
                "stats": {"pending": 1},
            },
        },
    )
    result = live_data.fetch_vendor_bids(_jwt())
    assert result["stats"]["pending"] == 1


def test_fetch_vendor_subscription_null(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": None},
    )
    assert live_data.fetch_vendor_subscription(_jwt()) is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_fetch_raises_on_success_false(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": False, "message": "unauthorized"},
    )
    with pytest.raises(LiveDataFetchError, match="unauthorized"):
        live_data.fetch_my_bookings(_jwt())


def test_fetch_raises_on_transport_error(monkeypatch):
    _mock_httpx(
        monkeypatch,
        raises=httpx.HTTPError("network down"),
    )
    with pytest.raises(LiveDataFetchError, match="network down"):
        live_data.fetch_my_bookings(_jwt())


def test_fetch_raises_on_non_dict_response(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body=["list", "not", "dict"],
    )
    with pytest.raises(LiveDataFetchError):
        live_data.fetch_my_bookings(_jwt())


def test_fetch_raises_on_http_500(monkeypatch):
    _mock_httpx(
        monkeypatch,
        status_code=500,
        body={"error": "server error"},
    )
    with pytest.raises(LiveDataFetchError, match="500"):
        live_data.fetch_my_bookings(_jwt())


def test_fetch_raises_on_http_401(monkeypatch):
    _mock_httpx(
        monkeypatch,
        status_code=401,
        body={"error": "unauthorized"},
    )
    with pytest.raises(LiveDataFetchError, match="401"):
        live_data.fetch_my_bookings(_jwt())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_fetch_for_intent_requires_jwt():
    with pytest.raises(LiveDataFetchError, match="JWT"):
        live_data.fetch_for_intent(
            LiveDataIntent.MY_BOOKINGS, user_jwt=None
        )


def test_fetch_for_intent_unknown_intent():
    with pytest.raises(LiveDataFetchError, match="no fetcher"):
        live_data.fetch_for_intent(
            LiveDataIntent.NONE, user_jwt="x"
        )


def test_fetch_for_intent_dispatch(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": {"bookings": [], "total": 0}},
    )
    label, payload = live_data.fetch_for_intent(
        LiveDataIntent.MY_BOOKINGS, user_jwt=_jwt()
    )
    assert label == "GET /api/bookings/my"
    assert payload["total"] == 0
