"""Format live-data payloads into compact text for the LLM.

Hard rules:
  * At most 5 items per list. Never the full set.
  * Truncate strings to prevent prompt bloat.
  * Never include raw JSON with internal fields (bank accounts, PANs,
    phone numbers). Only user-facing fields the LLM needs to answer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.capabilities.support.live_data_classifier import LiveDataIntent

_MAX_ITEMS = 5
_MAX_STR = 120


def _s(value: Any, max_len: int = _MAX_STR) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


def _date(value: Any) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).strftime("%d %b %Y")
    except ValueError:
        return str(value)[:10]


def _money(amount: Any, currency: str | None = None) -> str:
    """Format a money amount.

    When ``currency`` is supplied, prefix the amount with it
    (``INR 1,000.00``). Negative amounts put the sign first
    (``-INR 140.00``). No symbol is assumed if currency is None.
    """
    if amount is None or amount == "":
        return "—"
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    sign = "-" if value < 0 else ""
    numeric = f"{abs(value):,.2f}"
    if currency:
        return f"{sign}{str(currency).strip().upper()} {numeric}"
    return f"{sign}{numeric}"


# Per-intent formatters

def _fmt_my_bookings(payload: dict[str, Any]) -> str:
    bookings = payload.get("bookings", [])[:_MAX_ITEMS]
    total = payload.get("total", len(bookings))
    if not bookings:
        return "You have no bookings."

    lines = [f"YOUR BOOKINGS (most recent {len(bookings)} of {total}):"]
    for b in bookings:
        destination = (
            (b.get("package") or {}).get("name")
            or (b.get("metadata") or {}).get("destination")
            or (b.get("metadata") or {}).get("enquiryTitle")
            or (b.get("bookingType") or "").capitalize()   # "package" / "vehicle"
            or "—"
        )
        lines.append(
            f"- {_s(b.get('bookingCode'))} | {_s(destination)} | "
            f"{_date(b.get('startDate'))} → {_date(b.get('endDate'))} | "
            f"{_money(b.get('amount'), b.get('currency') or 'INR')} | "
            f"status={_s(b.get('status'))} | "
            f"payment={_s(b.get('paymentStatus'))}"
        )
    return "\n".join(lines)

def _fmt_my_booking_detail(payload: dict[str, Any]) -> str:
    booking = payload.get("booking")
    if not isinstance(booking, dict) or not booking:
        return "The booking endpoint returned no booking details."

    package = booking.get("package")
    if not isinstance(package, dict):
        package = {}

    metadata = booking.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    payment = booking.get("payment")
    if not isinstance(payment, dict):
        payment = {}

    destination = (
        package.get("name")
        or metadata.get("destination")
        or metadata.get("enquiryTitle")
    )
    payment_status = booking.get("paymentStatus") or payment.get("status")
    currency = booking.get("currency") or payment.get("currency") or "INR"
    amount = booking.get("amount")
    if amount is None:
        amount = payment.get("amount")

    lines = [
        "BOOKING DETAILS:",
        f"- Booking code: {_s(booking.get('bookingCode'))}",
        f"- Package or destination: {_s(destination)}",
        f"- Booking type: {_s(booking.get('bookingType'))}",
        f"- Start date: {_date(booking.get('startDate'))}",
        f"- End date: {_date(booking.get('endDate'))}",
        f"- Booking status: {_s(booking.get('status'))}",
        f"- Payment status: {_s(payment_status)}",
        f"- Amount: {_money(amount, currency)}",
    ]

    return "\n".join(lines)

def _fmt_my_payments(payload: dict[str, Any]) -> str:
    payments = payload.get("payments", [])[:_MAX_ITEMS]
    total = payload.get("total", len(payments))
    if not payments:
        return "You have no payments recorded."

    lines = [f"YOUR PAYMENTS (most recent {len(payments)} of {total}):"]
    for p in payments:
        amount = p.get("totalAmount") or p.get("amount")
        lines.append(
            f"- {_date(p.get('createdAt'))} | "
            f"{_money(amount, p.get('currency') or 'INR')} | "
            f"status={_s(p.get('status'))} | "
            f"type={_s(p.get('paymentType'))} | "
            f"method={_s(p.get('paymentMethod'))}"
        )
    return "\n".join(lines)


def _fmt_my_coins(payload: dict[str, Any]) -> str:
    return (
        f"YOUR SWABI COINS:\n"
        f"- Available balance: {_s(payload.get('coinBalance'), 20)} coins\n"
        f"- Total (incl. pending): "
        f"{_s(payload.get('totalCoinsBalance'), 20)} coins\n"
        f"- Pending: {_s(payload.get('pendingCoins'), 20)} coins\n"
        f"- Lifetime savings: {_s(payload.get('lifetimeSavings'), 20)}\n"
        f"- Credit balance: {_s(payload.get('creditBalance'), 20)}"
    )


def _fmt_my_coin_history(payload: dict[str, Any]) -> str:
    txs = payload.get("transactions", [])[:_MAX_ITEMS]
    total = payload.get("total", len(txs))
    if not txs:
        return "You have no coin transactions yet."
    lines = [f"YOUR COIN TRANSACTIONS (most recent {len(txs)} of {total}):"]
    for t in txs:
        lines.append(
            f"- {_date(t.get('createdAt'))} | {_s(t.get('type'))} | "
            f"{_s(t.get('amount'), 20)} coins | "
            f"{_s(t.get('description'))}"
        )
    return "\n".join(lines)


def _fmt_my_enquiries(payload: dict[str, Any]) -> str:
    enquiries = payload.get("enquiries", [])[:_MAX_ITEMS]
    total = payload.get("total", len(enquiries))
    if not enquiries:
        return "You have no enquiries."
    lines = [f"YOUR ENQUIRIES (most recent {len(enquiries)} of {total}):"]
    for e in enquiries:
        lines.append(
            f"- \"{_s(e.get('title'))}\" | {_s(e.get('destination'))} | "
            f"{_s(e.get('duration'))} | "
            f"{_s(e.get('travelers'), 20)} travellers | "
            f"budget {_money(e.get('budget'), 'INR')} | "
            f"status={_s(e.get('status'))} | "
            f"bids={_s(e.get('bidCount'), 5)}"
        )
    return "\n".join(lines)

def _fmt_my_enquiry_bids(payload: dict[str, Any]) -> str:
    bids = payload.get("bids", [])[:_MAX_ITEMS]
    total = payload.get("total", len(bids))
    if not bids:
        return f"BIDS ON YOUR ENQUIRY: none yet (total {total})."

    lines = [
        f"BIDS ON YOUR ENQUIRY (most recent {len(bids)} of {total}):"
    ]
    for b in bids:
        agent = b.get("agent") or {}
        lines.append(
            f"- {_money(b.get('amount'), b.get('currency') or 'INR')} "
            f"from {_s(agent.get('companyName'))} | "
            f"status={_s(b.get('status'))} | "
            f"negotiation={_s(b.get('negotiationStatus'))} | "
            f"created={_date(b.get('createdAt'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_wallet(payload: dict[str, Any]) -> str:
    return (
        "YOUR VENDOR WALLET:\n"
        f"- Withdrawable (escrow): "
        f"{_money(payload.get('escrowBalance'), 'INR')}\n"
        f"- Top-up balance (bids only): "
        f"{_money(payload.get('topupBalance'), 'INR')}\n"
        f"- Locked (pending withdrawal): "
        f"{_money(payload.get('lockedBalance'), 'INR')}\n"
        f"- Total withdrawable: "
        f"{_money(payload.get('withdrawable'), 'INR')}\n"
        f"- Available for bids: "
        f"{_money(payload.get('availableForBids'), 'INR')}\n"
        f"- Lifetime earned: "
        f"{_money(payload.get('lifetimeEarned'), 'INR')}\n"
        f"- Lifetime withdrawn: "
        f"{_money(payload.get('lifetimeWithdrawn'), 'INR')}\n"
        f"- Lifetime bid spend: "
        f"{_money(payload.get('lifetimeBidSpend'), 'INR')}\n"
        f"- Lifetime top-up: "
        f"{_money(payload.get('lifetimeTopup'), 'INR')}"
    )


def _fmt_vendor_transactions(payload: dict[str, Any]) -> str:
    txs = payload.get("transactions", [])[:_MAX_ITEMS]
    total = payload.get("total", len(txs))
    if not txs:
        return "No wallet transactions yet."
    lines = [f"YOUR WALLET LEDGER (most recent {len(txs)} of {total}):"]
    for t in txs:
        lines.append(
            f"- {_date(t.get('createdAt'))} | {_s(t.get('type'))} | "
            f"{_money(t.get('amount'), 'INR')} | "
            f"{_s(t.get('description'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_withdrawals(payload: dict[str, Any]) -> str:
    ws = payload.get("withdrawals", [])[:_MAX_ITEMS]
    total = payload.get("total", len(ws))
    if not ws:
        return "You have no withdrawal requests."
    lines = [f"YOUR WITHDRAWALS (most recent {len(ws)} of {total}):"]
    for w in ws:
        lines.append(
            f"- {_date(w.get('createdAt'))} | "
            f"{_money(w.get('amount'), 'INR')} | "
            f"status={_s(w.get('status'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_inquiries(payload: dict[str, Any]) -> str:
    inquiries = payload.get("inquiries", [])[:_MAX_ITEMS]
    total = payload.get("total", len(inquiries))
    stats = payload.get("stats", {})
    if not inquiries:
        return "You have no enquiries to bid on."
    lines = [
        f"ENQUIRIES YOU CAN BID ON (most recent {len(inquiries)} of "
        f"{total}; stats: new={stats.get('new', 0)}, "
        f"bid_placed={stats.get('bid_placed', 0)}, "
        f"won={stats.get('won', 0)}, lost={stats.get('lost', 0)}):"
    ]
    for e in inquiries:
        dates = e.get("travelDates") or {}
        lines.append(
            f"- {_s(e.get('inquiryCode'))} | {_s(e.get('destination'))} | "
            f"{_date(dates.get('from'))} → {_date(dates.get('to'))} | "
            f"{_s(e.get('persons'), 5)} travellers | "
            f"budget {_money(e.get('budget'), 'INR')} | "
            f"status={_s(e.get('status'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_bookings(payload: dict[str, Any]) -> str:
    bookings = payload.get("bookings", [])[:_MAX_ITEMS]
    total = payload.get("total", len(bookings))
    if not bookings:
        return "You have no vendor bookings."
    lines = [
        f"YOUR VENDOR BOOKINGS (most recent {len(bookings)} of {total}):"
    ]
    for b in bookings:
        destination = (
            (b.get("metadata") or {}).get("destination")
            or (b.get("package") or {}).get("name")
            or "—"
        )
        lines.append(
            f"- {_s(b.get('bookingCode'))} | {_s(destination)} | "
            f"{_date(b.get('startDate'))} → {_date(b.get('endDate'))} | "
            f"{_money(b.get('amount'), b.get('currency') or 'INR')} | "
            f"status={_s(b.get('status'))} | "
            f"payment={_s(b.get('paymentStatus'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_bids(payload: dict[str, Any]) -> str:
    bids = payload.get("bids", [])[:_MAX_ITEMS]
    total = payload.get("total", len(bids))
    stats = payload.get("stats", {})
    if not bids:
        return "You have no bids."
    lines = [
        f"YOUR BIDS (most recent {len(bids)} of {total}; "
        f"stats: pending={stats.get('pending', 0)}, "
        f"accepted={stats.get('accepted', 0)}, "
        f"rejected={stats.get('rejected', 0)}):"
    ]
    for b in bids:
        enq = b.get("enquiry") or {}
        lines.append(
            f"- {_money(b.get('amount'), 'INR')} on "
            f"\"{_s(enq.get('title'))}\" "
            f"({_s(enq.get('destination'))}) | "
            f"status={_s(b.get('status'))} | "
            f"negotiation={_s(b.get('negotiationStatus'))} | "
            f"created={_date(b.get('createdAt'))}"
        )
    return "\n".join(lines)


def _fmt_vendor_subscription(payload: Any) -> str:
    if not payload:
        return "You have no active subscription plan."
    if isinstance(payload, dict):
        plan = (
            payload.get("plan")
            or payload.get("planName")
            or "unknown plan"
        )
        status = payload.get("status", "—")
        expires = payload.get("expiresAt") or payload.get("endDate")
        return (
            f"YOUR SUBSCRIPTION:\n"
            f"- Plan: {_s(plan)}\n"
            f"- Status: {_s(status)}\n"
            f"- Expires: {_date(expires)}"
        )
    return f"Your subscription: {_s(payload)}"



# Dispatch
_FORMATTERS = {
    LiveDataIntent.MY_BOOKINGS: _fmt_my_bookings,
    LiveDataIntent.MY_BOOKING_DETAIL: _fmt_my_booking_detail,
    LiveDataIntent.MY_PAYMENTS: _fmt_my_payments,
    LiveDataIntent.MY_COINS: _fmt_my_coins,
    LiveDataIntent.MY_COIN_HISTORY: _fmt_my_coin_history,
    LiveDataIntent.MY_ENQUIRIES: _fmt_my_enquiries,
    LiveDataIntent.VENDOR_WALLET: _fmt_vendor_wallet,
    LiveDataIntent.VENDOR_TRANSACTIONS: _fmt_vendor_transactions,
    LiveDataIntent.VENDOR_WITHDRAWALS: _fmt_vendor_withdrawals,
    LiveDataIntent.VENDOR_INQUIRIES: _fmt_vendor_inquiries,
    LiveDataIntent.VENDOR_BOOKINGS: _fmt_vendor_bookings,
    LiveDataIntent.VENDOR_BIDS: _fmt_vendor_bids,
    LiveDataIntent.VENDOR_SUBSCRIPTION: _fmt_vendor_subscription,
    LiveDataIntent.MY_ENQUIRY_BIDS: _fmt_my_enquiry_bids,
}


def format_for_intent(intent: LiveDataIntent, payload: dict[str, Any]) -> str:
    fmt = _FORMATTERS.get(intent)
    if fmt is None:
        return "Live data available but not formatted."
    return fmt(payload)
