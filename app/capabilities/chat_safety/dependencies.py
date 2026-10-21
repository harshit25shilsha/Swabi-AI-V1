"""FastAPI dependencies for the Chat Safety route.

The service-token check now lives in ``app.core.service_auth`` so it
can be shared across service-to-service routes. This module re-exports
it for backward compatibility with existing imports.
"""

from __future__ import annotations

from app.core.service_auth import verify_service_token

__all__ = ["verify_service_token", "verify_token"]

# Backend-compatible alias.
verify_token = verify_service_token
