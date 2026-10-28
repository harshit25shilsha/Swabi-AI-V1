"""Async HTTP client for the Swabi NestJS backend."""

from __future__ import annotations

import asyncio
import json

import httpx

from app.core.auth_context import get_user_jwt
from app.core.config import settings
from app.core.exceptions import SwabiAPIError, SwabiAuthenticationError
from app.core.logging import get_logger, request_id_var
from app.swabi.crypto import SwabiDecryptionError, decrypt_payload

log = get_logger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SwabiClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or settings.swabi_api_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.swabi_api_key
        self.timeout = timeout or settings.swabi_api_timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": request_id_var.get(),
        }
        user_jwt = get_user_jwt()
        if user_jwt:
            headers["Authorization"] = f"Bearer {user_jwt}"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, method: str, path: str, **kwargs: object) -> dict:
        url = f"{self.base_url}{path}"
        attempt = 0
        delay = 0.5

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while attempt <= self.max_retries:
                attempt += 1
                try:
                    resp = await client.request(method, url, headers=self._headers(), **kwargs)
                except httpx.TimeoutException as exc:
                    if attempt > self.max_retries:
                        raise SwabiAPIError(f"Swabi timeout: {url}") from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                except httpx.HTTPError as exc:
                    if attempt > self.max_retries:
                        raise SwabiAPIError(f"Swabi network error: {exc}") from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if resp.status_code == 401:
                    raise SwabiAuthenticationError("Invalid Swabi credentials")
                if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if resp.status_code >= 400:
                    raise SwabiAPIError(
                        f"Swabi HTTP {resp.status_code} on {url}: {resp.text[:300]}"
                    )

                try:
                    data = resp.json()
                except ValueError as exc:
                    raise SwabiAPIError(f"Swabi returned non-JSON: {exc}") from exc

                # Decrypt if the backend returns an encrypted envelope.
                if (
                    isinstance(data, dict)
                    and set(data.keys()) == {"payload"}
                    and settings.swabi_response_passphrase
                ):
                    log.info(
                        "swabi_decrypt",
                        extra={
                            "path": path,
                            "ciphertext_len": len(data["payload"]),
                        },
                    )
                    try:
                        plaintext = decrypt_payload(
                            data["payload"],
                            settings.swabi_response_passphrase,
                            kdf=settings.swabi_response_kdf,
                        )
                    except SwabiDecryptionError as exc:
                        raise SwabiAPIError(
                            f"Swabi decryption failed on {path}: {exc}"
                        ) from exc
                    try:
                        data = json.loads(plaintext)
                    except json.JSONDecodeError as exc:
                        raise SwabiAPIError(
                            f"decrypted payload is not JSON on {path}: {exc}"
                        ) from exc

                if not isinstance(data, dict):
                    raise SwabiAPIError(
                        f"Swabi response is not a dict on {path}: "
                        f"{type(data).__name__}"
                    )
                return data

        raise SwabiAPIError(f"Swabi request failed after retries: {url}")

    # --- Public methods (Phase 0 keeps these minimal) ----------------------

    async def get_customer_profile(self, user_id: str) -> dict:
        return await self._request("GET", f"/customers/{user_id}/profile")

    async def search_packages(self, query: str, **params: object) -> dict:
        return await self._request("GET", "/packages/search", params={"q": query, **params})

    async def search_activities(self, query: str, **params: object) -> dict:
        return await self._request("GET", "/activities/search", params={"q": query, **params})
