# Phase: Guardrail Integration — Final Report

**Branch:** `feat/guardrail-integration`
**Scope:** Integrate the Guardrail Service (GDv1) into the Swabi AI
Foundation (Phase 0) as a dedicated Chat Safety capability.
**Status:** Integration complete + post-integration hardening. Production
release gates remain open.
**Date:** October 2026

This report follows the structure required by §10 of
`docs/guardrail_integration_plan.md`.

---

## 1. What was inspected

### Phase 0 — AI Foundation

- `app/main.py`, `app/core/{config,logging,exceptions,redis,auth_context}.py`
- `app/llm/{gateway,groq,schemas}.py`
- `app/swabi/{client,schemas}.py`, `app/swabi/tools/base.py`
- `app/rag/interface.py`, `app/graph/base.py`, `app/evaluation/`
- `tests/` — 23 passing tests at baseline

### GDv1 — Guardrail Service

- `app/api/{moderation,metrics}.py`
- `app/config.py`, `app/models/moderation.py`
- `app/detectors/` — 10 detector modules
- `app/llm/{providers,moderator}.py`
- `app/policies/policy.py`, `app/services/moderation.py`
- `app/observability/{counters,circuit_breaker,middleware,logging_setup}.py`
- `tests/` — 17 test files

### Overlap and asymmetry

The two services sit on opposite sides of the auth boundary:

| Concern | Phase 0 | GDv1 |
|---|---|---|
| Caller | Frontend | NestJS |
| Inbound auth | User JWT | Service token |
| Acts as user? | Yes | No |
| Talks to Swabi APIs? | Yes | No |

The integration preserves this asymmetry. In particular, the guardrail
does not receive or forward user JWTs.

---

## 2. What was reused unchanged

### From GDv1

- All 10 original deterministic detectors — verbatim, byte-for-byte
  identical detection logic.
- `policies/policy.py` — `BLOCKED_CATEGORIES` and `apply_policy`,
  unchanged. The separation between "detect evidence" and "apply
  policy" is preserved.
- `SYSTEM_PROMPT` and `PROMPT_VERSION` — byte-identical at the time of
  the move. The scattered-number hardening extended the prompt
  afterward (see Addendum).
- Public response schema fields and `ALLOW`/`BLOCK` semantics.
- The moderation log event shape (fields, names, values).
- The `/metrics` endpoint shape and metric names.
- The circuit-breaker state machine.
- The fail-open behavior on provider outage.

### From Phase 0

- Config module structure (extended, not replaced).
- Exception hierarchy (extended).
- The request-ID contextvar (`request_id_var`).
- `SwabiClient`, `LLMGateway`, `BaseSwabiTool`, RAG interface, LangGraph
  base, evaluation runner.
- Middleware framework.

---

## 3. What changed

### Added

| File | Purpose |
|---|---|
| `app/capabilities/__init__.py` | Package marker |
| `app/capabilities/chat_safety/` | Full capability |
| `app/capabilities/chat_safety/llm_adapter.py` | Groq via `LLMGateway` + Gemini via `google-genai` |
| `app/api/routes/{metrics,chat_safety}.py` | Routes moved from GDv1 |
| `app/observability/{counters,circuit_breaker,middleware}.py` | Moved from GDv1 |
| `tests/unit/chat_safety/` | Unit-test suite |
| `tests/integration/chat_safety/` | Route-level tests |
| `tests/contract/test_chat_safety_contract.py` | Locks the public API shape |
| `tests/integration/test_auth_boundary.py` | Guards the JWT/service-token separation |
| `tests/integration/test_middleware_order.py` | Verifies middleware order |
| `docs/*` | This set of documents |

### Modified

| File | Change |
|---|---|
| `app/core/config.py` | Guardrail fields; property aliases; consensus default = 1; `LLM_MAX_TOKENS=512` |
| `app/core/logging.py` | Merged GDv1's JSON formatter; `setup_logging(level, fmt)` |
| `app/core/exceptions.py` | `GuardrailError`, `GuardrailUnavailableError`, `GuardrailValidationError`, `CircuitOpenError` |
| `app/main.py` | Middleware order; `USER_JWT_ROUTE_PREFIXES`; three routers |
| `app/observability/middleware.py` | Reads request_id from shared contextvar; does not write `X-Request-ID` |
| `app/api/routes/health.py` | Superset of Phase 0 and GDv1 health shapes |
| `app/capabilities/chat_safety/dependencies.py` | `secrets.compare_digest` |
| `app/llm/groq.py`, `app/llm/gateway.py` | Added `generate_sync()`; shared sync client; timeout per request; `json_validate_failed` fallback |
| `app/capabilities/chat_safety/moderator.py` | Uses adapter; raises `CircuitOpenError` |
| `requirements.txt` | Removed `groq`; added `google-genai` |

### Removed

| File | Reason |
|---|---|
| `app/capabilities/chat_safety/providers.py` | Replaced by `llm_adapter.py` |
| `groq` package dependency | Groq reached via shared gateway (httpx) |

### Test-suite corrections

- `tests/moderation.py` → `test_moderation_service.py`.
- `tests/test_number_words.py` — duplicate import removed; overlapping
  tests merged.
- `tests/test_disguised_numbers.py` — no-op `assert ... or True` deleted.
- `tests/test_integration_llm.py` — duplicate entries removed.
- `tests/conftest.py` — loads `.env` before setting dummy env vars.

---

## 4. Architecture after integration

Request flow for `POST /api/v1/validate-message`:

```
NestJS
  │ Authorization: Bearer <GUARDRAIL_API_TOKEN>
  ▼
request_id_middleware           (outermost)
  ▼
auth_context_middleware         (clears user_jwt_var for this route)
  ▼
ObservabilityMiddleware         (innermost)
  ▼
chat_safety route
  ├─ verify_token (compare_digest)
  ├─ ChatSafetyService.moderate()
  │    ├─ normalize
  │    ├─ hard detectors (bool)
  │    ├─ soft detectors (ALLOW / REVIEW / BLOCK)
  │    ├─ if no block: ChatSafetyLLMAdapter.classify()
  │    │     └─ LLMGateway.generate_sync() with provider ladder
  │    ├─ apply_policy
  │    └─ escalation: soft REVIEW + LLM ALLOW -> source="review"
  └─ return ModerationResponse
  ▼
ObservabilityMiddleware emits moderation log line + counters
  ▼
request_id_middleware sets X-Request-ID on the response
  ▼
NestJS receives ModerationResponse
```

Component responsibilities:

- **Route:** validate request, authenticate, invoke service, serialize.
- **Service:** orchestrate normalize → detect → LLM → policy → escalate.
- **Detectors (hard):** cheap deterministic rules.
- **Detectors (soft):** aggregate rules; can produce REVIEW.
- **Moderator:** provider ladder, circuits, JSON parsing, consensus.
- **Adapter:** provider name + prompt → raw response string.
- **LLM gateway:** provider transport (httpx + retries + pooling).
- **Policy:** evidence → final decision.
- **Middleware:** correlation, counters, log event.

---

## 5. API compatibility

### Public endpoint

`POST /api/v1/validate-message` — unchanged path, unchanged method.

### Request body

Unchanged:

```json
{
  "message": "…",
  "message_id": "…",
  "conversation_id": "…",
  "sender_id": "…",
  "sender_role": "user" | "vendor"
}
```

Only `message` is required.

### Response body

Unchanged field set, locked by
`tests/contract/test_chat_safety_contract.py`:

```json
{
  "allowed": bool,
  "action": "ALLOW" | "BLOCK",
  "category": string | null,
  "confidence": float,
  "source": "deterministic" | "llm" | "review" | "fail_open",
  "request_id": "gr_…",
  "prompt_version": "marketplace_…",
  "provider": "groq" | "gemini" | ""
}
```

`source="review"` was added post-integration. It is additive: existing
consumers that only check `allowed` and `action` require no change.

### Auth

Unchanged: `Authorization: Bearer <GUARDRAIL_API_TOKEN>`.
Tightened: comparison uses `secrets.compare_digest`.

### Internal additions not exposed

- `GuardrailDecision` and `GuardrailResult` — internal.
- `REVIEW` and `UNAVAILABLE` decision values — never serialized as
  `action`; REVIEW is mapped to `allowed=false, action=BLOCK` with
  `source="review"`.

NestJS requires **no code change**.

---

## 6. Security and failure behavior

### Auth boundary

`auth_context_middleware` captures the user JWT only for routes under
`USER_JWT_ROUTE_PREFIXES`. The guardrail route is not on that list.
Every other route explicitly clears `user_jwt_var`.

Verified by `tests/integration/test_auth_boundary.py`.

### Service-token comparison

`verify_token` uses `secrets.compare_digest` to avoid timing
side-channels.

### Failure policy

**Fail-open on provider outage is preserved** when no soft detector has
flagged REVIEW:

```json
{
  "allowed": true,
  "action": "ALLOW",
  "category": null,
  "confidence": 0.0,
  "source": "fail_open"
}
```

If a soft detector has flagged REVIEW, the fail-open path returns
`allowed=false, action=BLOCK, source="review"` — the LLM failure does not
suppress the detector signal.

### Circuit breaker fix

The `assert last_exc is not None` bug from GDv1 is fixed.
`_classify_with_fallback` now raises `CircuitOpenError("all provider
circuits are open")` when no provider call was attempted.

### Logging

- No raw message text is logged.
- No JWTs or service tokens are logged.
- The moderation event carries a truncated SHA-256 of the message.
- Request-id correlation is preserved end-to-end.

### No database access

The AI service has no database client. Import-guard test scoped to
`app/capabilities/chat_safety/` ensures no DB library is pulled in.

---

## 7. Test results

Commands were run on the `feat/guardrail-integration` branch on Windows
with Python 3.12.10.

### Baseline before integration

```
pytest                      → 23 passed
```

### During integration (after each step)

| Step | Command | Result |
|---|---|---|
| 2 — config | `pytest` | 32 passed |
| 3 — logging | `pytest` | 48 passed |
| 4 — exceptions | `pytest` | 57 passed |
| 5 — observability | `pytest` | 83 passed |
| 7 — chat_safety move | `pytest` (after fixes) | 181 passed, 26 deselected |
| 8 — GuardrailResult | `pytest` | ~192 passed, 26 deselected |
| 9 — LLM adapter | `pytest` | ~206 passed, 26 deselected |
| 10 — wiring | `pytest` | ~236 passed, 26 deselected |
| 10 — wiring | `pytest -m integration -v` | 26 passed |

### Post-integration hardening

- `tests/unit/chat_safety/test_scattered_numbers.py` — new
- `tests/unit/test_groq_provider.py` — new (request shape: timeout)
- `tests/unit/chat_safety/test_moderation_service.py` — REVIEW
  escalation cases added
- `tests/integration/chat_safety/test_integration_llm.py` — scattered-
  numbers case added

### Lint and formatting

```
ruff check .                → All checks passed
ruff format --check .       → all files formatted
```

### Known failures at time of reporting

None.

### Skipped / deselected

The integration tests are excluded from the default `pytest` run via
`addopts = "-m 'not integration'"` in `pyproject.toml`.

---

## 8. Configuration changes

### New environment variables

```
GUARDRAIL_API_TOKEN
GEMINI_API_KEY
GEMINI_MODEL
GEMINI_TIMEOUT_SECONDS
LLM_FALLBACK_ENABLED
LLM_CONSENSUS_ATTEMPTS
LLM_CIRCUIT_FAILURE_THRESHOLD
LLM_CIRCUIT_OPEN_SECONDS
LLM_TEMPERATURE
LLM_MAX_TOKENS
SERVICE_VERSION
LOG_FORMAT
GROQ_CHAT_URL
```

### Values with non-default status

- `GROQ_API_KEY` — required, no default.
- `GUARDRAIL_API_TOKEN` — required, no default.
- `LLM_CONSENSUS_ATTEMPTS` — default `1`.
- `LLM_MAX_TOKENS` — default `512`.
- `LLM_FALLBACK_ENABLED` — default `true`.
- `GEMINI_TIMEOUT_SECONDS` — default `15`.

No secrets are committed. `.env.example` documents every variable.

---

## 9. Remaining limitations and open decisions

### Deliberately deferred

- **`Guardrail` ABC** — not introduced. When a second capability needs
  the shape, promote `GuardrailResult` to `app/guardrails/contracts.py`.
- **LangGraph integration for Chat Safety** — called directly.
- **Async conversion of the Chat Safety chain** — sync path preserved
  via `LLMGateway.generate_sync()`.
- **Async connection pooling** — only the sync path pools today.

### Production release gates (unchanged from plan §7.2)

- [ ] Fail-open policy approved in writing.
- [ ] Token rotation procedure documented and rehearsed once on staging.
- [ ] Deployment worker count decided and documented.
- [ ] Rate-limit expectation confirmed with the NestJS team.
- [ ] Manager sign-off on the failure policy and the auth boundary.

### Known operational characteristics

- `/metrics` and `/health` are unauthenticated. Bind behind a reverse
  proxy that restricts inbound access.
- Counters and circuit state are in-memory per process.
- The Gemini fallback is configured but has not been exercised against
  the live API in CI. Mocked tests cover the dispatch path.
- Tail latency: benchmark showed median ~890ms and P95 ~4s on the sync
  path. The tail is a policy question (bound the wait vs. accept it),
  not a bug.

---

## 10. Next recommended phase

**Phase 1 — AI Enquiry Understanding**, on top of the now-unified
foundation. Concretely:

- `POST /enquiry/understand` — natural language → structured enquiry.
- Uses `LLMGateway.generate_structured()` with a Pydantic
  `EnquiryExtraction` schema.
- Missing-field detection and clarification list.
- Reads customer / family context via `SwabiClient` when authorized
  (JWT forwarded through `USER_JWT_ROUTE_PREFIXES`).
- Adds tests under `tests/unit/enquiry/` and
  `tests/integration/enquiry/`.
- Reuses the shared config, logging, exceptions, and observability.

The foundation is ready. Chat Safety demonstrates the pattern: a
capability directory, a service class, a thin route, and contract
guards. Phase 1 follows the same shape.

---

## Addendum — Post-integration hardening

Changes made after the initial report, on the same branch, prior to
Phase 1.

### Scattered-number detection

Added a deterministic detector for phone numbers hidden across natural
language. The reported case (*"ek room, do bed, teen raat, chaar log,
paach bje phuchenge, ..."*) previously passed as ALLOW because every
individual number looked innocent in isolation and none of the existing
detectors saw a chain.

- New file: `app/capabilities/chat_safety/detectors/scattered_numbers.py`
- New tri-state contract: `GuardrailDecision.ALLOW / REVIEW / BLOCK`
- New `Action.REVIEW` value accepted from the LLM and passed through
  `apply_policy`
- Service layer escalates `soft REVIEW + LLM ALLOW` to `source="review"`
  (public response: `allowed=false, action=BLOCK`)
- Prompt extended with a "SCATTERED NUMBERS" section instructing the
  classifier to consider aggregate patterns and to return `REVIEW` when
  uncertain

### Groq provider hardening

- Reduced `LLM_MAX_TOKENS` default from 1024 to **512**.
  - 128 was tried and rejected: reasoning models consume 80–150 tokens
    on internal reasoning before emitting any output, causing Groq to
    return `json_validate_failed` with empty `failed_generation`.
- Added a fallback in both `generate()` and `generate_sync()`: on 400
  with `json_validate_failed`, retry once without
  `response_format={"type":"json_object"}`.
- Passed `timeout=self.timeout` per request to the shared sync client
  so a custom provider timeout overrides the module-level default.

### Shared sync HTTP client

- Module-level `httpx.Client` with pooling
  (`max_keepalive_connections=20`, `max_connections=50`,
  `keepalive_expiry=30s`).
- Lazy init with a lock; closed on FastAPI shutdown and via `atexit`.
- Retry delay unified at 0.2s across async and sync paths.
- Refactored `groq.py` so both paths share `_build_payload`,
  `_headers`, `_is_json_validate_failure`, and `_parse_response`.

### Deferred

- Async connection pooling — only the sync path pools today. Deferred
  to when a second LLM-backed capability lands (Phase 1) or when load
  justifies it.
- Tail latency policy — benchmark showed median ~890ms, P95 ~4s. The
  tail is a policy question (bound the wait vs accept it), not a bug.
  Not decided in this phase.

### Test-suite impact

- `tests/unit/chat_safety/test_scattered_numbers.py` — new
- `tests/unit/test_groq_provider.py` — new (request shape: timeout)
- `tests/unit/chat_safety/test_moderation_service.py` — REVIEW
  escalation cases added
- `tests/integration/chat_safety/test_integration_llm.py` —
  scattered-numbers case added to `DISGUISED_MUST_BLOCK`

The public response contract (field set, `ALLOW` / `BLOCK` semantics) is
unchanged. The `source` field gained a new value (`"review"`), which is
additive.