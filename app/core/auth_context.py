"""Request-scoped auth context.

Holds the caller's JWT for the duration of a request. Set by the
middleware in `app/main.py` and consumed by `SwabiClient`.

Rules:
    * The token is opaque here. It is never decoded, validated, or logged.
    * NestJS remains the only authority for authentication and authorization.
    * If no user token is present, SwabiClient falls back to the service key.
"""

from __future__ import annotations

from contextvars import ContextVar

# Caller-forwarded user token (opaque). Set from `Authorization: Bearer ...`.
user_jwt_var: ContextVar[str | None] = ContextVar("user_jwt", default=None)

# Optional caller-supplied user id (used only for logging correlation).
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_user_jwt(token: str | None) -> None:
    user_jwt_var.set(token)


def get_user_jwt() -> str | None:
    return user_jwt_var.get()


def set_user_id(user_id: str | None) -> None:
    user_id_var.set(user_id)


def get_user_id() -> str | None:
    return user_id_var.get()
