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
    assert record[0]["params"] == {}


def test_fetch_my_payments_with_status_filter(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [{"id": "p1"}]},
        record=record,
    )

    live_data.fetch_my_payments(_jwt(), status="completed")

    assert record[0]["params"] == {"status": "completed"}


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


def test_fetch_retries_timeout_and_succeeds(monkeypatch):
    calls = 0

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None, params=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("slow first attempt")
            return _FakeResponse(
                200,
                {"success": True, "data": {"bookings": [], "total": 0}},
            )

    monkeypatch.setattr(
        "app.capabilities.support.live_data.httpx.Client", FakeClient
    )
    monkeypatch.setattr("app.capabilities.support.live_data.time.sleep", lambda _: None)

    result = live_data.fetch_my_bookings(_jwt())

    assert result["total"] == 0
    assert calls == 2


def test_fetch_raises_after_timeout_retry_exhausted(monkeypatch):
    calls = 0

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None, params=None):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("backend still slow")

    monkeypatch.setattr(
        "app.capabilities.support.live_data.httpx.Client", FakeClient
    )
    monkeypatch.setattr("app.capabilities.support.live_data.time.sleep", lambda _: None)

    with pytest.raises(LiveDataFetchError, match="timeout"):
        live_data.fetch_my_bookings(_jwt())

    assert calls == live_data.LIVE_FETCH_MAX_ATTEMPTS


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


def test_fetch_package_detail_by_code(monkeypatch):
    record: list = []
    package = {
        "packageCode": "PKG-922",
        "title": "Rishikesh Tours",
    }
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {"packages": [package], "total": 1},
        },
        record=record,
    )

    label, payload = live_data.fetch_for_intent(
        LiveDataIntent.PACKAGE_DETAIL,
        user_jwt=None,
        entities={"package_code": "PKG-922"},
    )

    assert label == "GET /api/packages (lookup by package code)"
    assert payload == {"package": package}
    assert record[0]["url"].endswith("/packages")
    assert record[0]["params"] == {"page": 1, "limit": 100}
    assert "Authorization" not in record[0]["headers"]


def test_fetch_package_detail_by_code_searches_later_pages(monkeypatch):
    requests: list[dict] = []
    pages = {
        1: {"packages": [{"packageCode": "PKG-921"}], "total": 2},
        2: {"packages": [{"packageCode": "PKG-922"}], "total": 2},
    }

    def fake_get(path, *, user_jwt, params=None):
        requests.append({"path": path, "params": params})
        return {"data": pages[params["page"]]}

    monkeypatch.setattr(live_data, "_get", fake_get)

    payload = live_data.fetch_package_by_code(None, "PKG-922")

    assert payload["package"]["packageCode"] == "PKG-922"
    assert [request["params"]["page"] for request in requests] == [1, 2]


def test_fetch_package_detail_by_code_reports_missing_package(monkeypatch):
    _mock_httpx(
        monkeypatch,
        body={
            "success": True,
            "data": {"packages": [], "total": 0},
        },
    )

    payload = live_data.fetch_package_by_code(None, "PKG-922")

    assert payload == {"package": {}, "not_found_code": "PKG-922"}


def test_fetch_for_payment_history_omits_unspecified_status(monkeypatch):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": [{"id": "p1"}]},
        record=record,
    )

    live_data.fetch_for_intent(
        LiveDataIntent.MY_PAYMENTS,
        user_jwt=_jwt(),
        entities={},
    )

    assert record[0]["params"] == {}


def test_fetch_latest_enquiry_bids_uses_newest_enquiry(monkeypatch):
    enquiries = [
        {
            "id": "11111111-1111-4111-8111-111111111111",
            "title": "Older trip",
            "createdAt": "2026-10-12T10:00:00Z",
        },
        {
            "id": "22222222-2222-4222-8222-222222222222",
            "title": "Newest trip",
            "createdAt": "2026-10-13T10:00:00Z",
        },
    ]
    requested_ids: list[str] = []
    monkeypatch.setattr(
        live_data,
        "fetch_my_enquiries",
        lambda user_jwt: {"enquiries": enquiries, "total": 2},
    )

    def fake_fetch_bids(user_jwt, enquiry_id):
        requested_ids.append(enquiry_id)
        return {"bids": [], "total": 0}

    monkeypatch.setitem(
        live_data._FETCHERS,
        LiveDataIntent.MY_ENQUIRY_BIDS,
        fake_fetch_bids,
    )

    label, payload = live_data.fetch_for_intent(
        LiveDataIntent.MY_ENQUIRY_BIDS,
        user_jwt=_jwt(),
        entities={},
    )

    assert requested_ids == ["22222222-2222-4222-8222-222222222222"]
    assert "GET /api/enquiries/my" in label
    assert payload["enquiry"]["title"] == "Newest trip"
    assert payload["total"] == 0
    from app.capabilities.support.live_data_format import format_for_intent

    answer = format_for_intent(LiveDataIntent.MY_ENQUIRY_BIDS, payload)
    assert "Newest trip" in answer
    assert "No bids have been received on" in answer
    assert "any of your enquiries" not in answer


def test_fetch_latest_enquiry_bids_handles_no_enquiries(monkeypatch):
    monkeypatch.setattr(
        live_data,
        "fetch_my_enquiries",
        lambda user_jwt: {"enquiries": [], "total": 0},
    )
    monkeypatch.setitem(
        live_data._FETCHERS,
        LiveDataIntent.MY_ENQUIRY_BIDS,
        lambda *_: pytest.fail("bids endpoint should not be called"),
    )

    label, payload = live_data.fetch_for_intent(
        LiveDataIntent.MY_ENQUIRY_BIDS,
        user_jwt=_jwt(),
        entities={},
    )

    assert label == "GET /api/enquiries/my"
    assert payload["enquiry"] is None


@pytest.mark.parametrize(
    "intent,status",
    [
        (LiveDataIntent.MY_BOOKINGS, "cancel"),
        (LiveDataIntent.MY_BOOKINGS, "canceled"),
        (LiveDataIntent.MY_BOOKINGS, "cancelled"),
        (LiveDataIntent.MY_BOOKINGS, "complete"),
        (LiveDataIntent.MY_BOOKINGS, "completed"),
        (LiveDataIntent.MY_PAYMENTS, "cancel"),
        (LiveDataIntent.MY_PAYMENTS, "canceled"),
        (LiveDataIntent.MY_PAYMENTS, "complete"),
    ],
)
def test_fetch_for_intent_normalizes_status_aliases(
    monkeypatch, intent, status
):
    record: list = []
    _mock_httpx(
        monkeypatch,
        body={"success": True, "data": {"bookings": [], "total": 0}},
        record=record,
    )

    live_data.fetch_for_intent(
        intent,
        user_jwt=_jwt(),
        entities={"status": status},
    )

    expected = (
        "cancelled" if status in {"cancel", "canceled", "cancelled"}
        else "completed"
    )
    assert record[0]["params"]["status"] == expected
