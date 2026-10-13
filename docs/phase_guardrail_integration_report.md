# Phase: Guardrail Integration — Final Report

**Branch:** `feat/guardrail-integration`
**Scope:** Integrate the Guardrail Service (GDv1) into the Swabi AI
Foundation (Phase 0) as a dedicated Chat Safety capability.
**Status:** Integration complete. Production release gates remain open.
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

- All 10 deterministic detectors — verbatim, byte-for-byte identical
  detection logic.
- `policies/policy.py` — `BLOCKED_CATEGORIES` and `apply_policy`,
  unchanged. The separation between "detect evidence" and "apply
  policy" is preserved.
- `SYSTEM_PROMPT` and `PROMPT_VERSION` — byte-identical. The prompt
  hash is unchanged, so any downstream audit correlation is preserved.
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
  base, evaluation runner — untouched by this phase.
- Middleware framework.

---

## 3. What changed

### Added

| File | Purpose |
|---|---|
| `app/capabilities/__init__.py` | Package marker |
| `app/capabilities/chat_safety/` | Full capability (detectors, policy, prompt, schemas, service, dependencies, adapter, moderator) |
| `app/capabilities/chat_safety/llm_adapter.py` | Groq via `LLMGateway` + Gemini via `google-genai` |
| `app/api/routes/{metrics,chat_safety}.py` | Routes moved from GDv1 |
| `app/observability/{counters,circuit_breaker,middleware}.py` | Moved from GDv1 to a shared location |
| `tests/unit/chat_safety/` | Full unit-test suite for the capability |
| `tests/integration/chat_safety/` | Route-level tests (mocked + live opt-in) |
| `tests/contract/test_chat_safety_contract.py` | Locks the public API shape |
| `tests/integration/test_auth_boundary.py` | Guards the JWT/service-token separation |
| `tests/integration/test_middleware_order.py` | Verifies actual middleware order |
| `docs/guardrail_integration_plan.md` | Plan (pre-implementation) |
| `docs/gdv1_test_verification.md` | Test-suite audit |
| `docs/architecture.md` | Reference doc |
| `docs/phase_guardrail_integration_report.md` | This report |

### Modified

| File | Change |
|---|---|
| `app/core/config.py` | Added guardrail fields; upper-case read-only property aliases; `LLM_CONSENSUS_ATTEMPTS` default = 1 |
| `app/core/logging.py` | Merged GDv1's JSON formatter; `setup_logging(level, fmt)`; request_id attached as a field |
| `app/core/exceptions.py` | Added `GuardrailError`, `GuardrailUnavailableError`, `GuardrailValidationError`, `CircuitOpenError` (subclass of `GuardrailUnavailableError`) |
| `app/main.py` | Three middlewares registered in LIFO order; `USER_JWT_ROUTE_PREFIXES` allow-list; three routers registered |
| `app/observability/middleware.py` | Reads request_id from shared contextvar; no longer writes `X-Request-ID` |
| `app/api/routes/health.py` | Superset of Phase 0 and GDv1 health shapes |
| `app/capabilities/chat_safety/dependencies.py` | `secrets.compare_digest` for token comparison |
| `app/llm/groq.py`, `app/llm/gateway.py` | Added `generate_sync()` — synchronous mirror for the sync Chat Safety chain |
| `app/capabilities/chat_safety/moderator.py` | Uses the adapter; raises `CircuitOpenError` when all circuits are open; no more `assert last_exc is not None` |
| `requirements.txt` | Removed `groq`; added `google-genai` |

### Removed

| File | Reason |
|---|---|
| `app/capabilities/chat_safety/providers.py` | Replaced by `llm_adapter.py` |
| `groq` package dependency | Groq now reached via the shared gateway (httpx) |

### Test-suite corrections (pre-move verification, per plan §5.10)

- `tests/moderation.py` → `test_moderation_service.py` (was not
  collected due to missing prefix).
- `tests/test_number_words.py` — duplicate import removed; two pairs of
  overlapping tests merged.
- `tests/test_disguised_numbers.py` — no-op `assert ... or True` test
  deleted.
- `tests/test_integration_llm.py` — duplicate entries removed from
  `NORMAL_MUST_ALLOW`.
- `tests/conftest.py` — loads `.env` before setting dummy env vars, so
  the developer's real Groq key is visible to integration tests.

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
  │    ├─ deterministic detectors
  │    ├─ if no match: ChatSafetyLLMAdapter.classify()
  │    │     └─ LLMGateway.generate_sync() with provider ladder
  │    ├─ apply_policy
  │    └─ return ModerationResponse
  ├─ attach request_id, prompt_version
  └─ stash result on request.state for the middleware to log
  ▼
ObservabilityMiddleware emits moderation log line + counters
  ▼
request_id_middleware sets X-Request-ID on the response
  ▼
NestJS receives ModerationResponse (public contract)
```

Component responsibilities:

- **Route:** validate request, authenticate, invoke service, serialize.
- **Service:** orchestrate normalize → detect → LLM → policy.
- **Detectors:** cheap deterministic rules.
- **Moderator:** provider ladder, circuits, JSON parsing, consensus.
- **Adapter:** turns a provider name + prompt into a raw response string.
- **LLM gateway:** provider transport (httpx + retries).
- **Policy:** converts LLM classification into a final decision.
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
  "source": "deterministic" | "llm" | "fail_open",
  "request_id": "gr_…",
  "prompt_version": "marketplace_…",
  "provider": "groq" | "gemini" | ""
}
```

### Auth

Unchanged: `Authorization: Bearer <GUARDRAIL_API_TOKEN>`.
Tightened: comparison now uses `secrets.compare_digest`.

### Internal additions not exposed

- `GuardrailDecision` and `GuardrailResult` — internal only.
- `REVIEW` and `UNAVAILABLE` decision values — never serialized on the
  public endpoint.

The NestJS backend requires **no code change**.

---

## 6. Security and failure behavior

### Auth boundary

`auth_context_middleware` captures the user JWT only for routes under
`USER_JWT_ROUTE_PREFIXES`. The guardrail route is not on that list.
Every other route explicitly clears `user_jwt_var`.

This is verified by `tests/integration/test_auth_boundary.py`, which
proves that a Bearer token on the guardrail route does not populate
`user_jwt_var`. The test will fail if someone later adds
`/api/v1/validate-message` to the allow-list.

### Service-token comparison

`verify_token` uses `secrets.compare_digest` to avoid timing
side-channels.

### Failure policy

**Fail-open on provider outage is preserved.** When all providers fail
or all circuits are open, the service returns:

```json
{
  "allowed": true,
  "action": "ALLOW",
  "category": null,
  "confidence": 0.0,
  "source": "fail_open"
}
```

This is a deliberate product decision, carried over from GDv1. It is
listed as a **production release gate** requiring explicit sign-off
(plan §7.2).

### Circuit breaker fix

The `assert last_exc is not None` bug from GDv1 is fixed.
`_classify_with_fallback` now raises `CircuitOpenError("all provider
circuits are open")` when no provider call was attempted, making the
intended `except CircuitOpenError` branch in `service.py` reachable.

### Logging

- No raw message text is logged.
- No JWTs or service tokens are logged.
- The moderation event carries a truncated SHA-256 of the message.
- The request-id correlation is preserved end-to-end.

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

The integration suite (opt-in, live Groq) was validated end-to-end in
step 8 and again after step 10. It passed both times once the `.env`
load-order fix was applied to `tests/conftest.py`.

### Lint and formatting

```
ruff check .                → All checks passed
ruff format --check .       → all files formatted
```

### Known failures at time of reporting

None. All failing tests encountered during the integration were fixed
within the step that introduced them.

### Skipped / deselected

The 26 integration tests are excluded from the default `pytest` run
via `addopts = "-m 'not integration'"` in `pyproject.toml`. They are
opt-in only and require a real `GROQ_API_KEY` in `.env`.

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
```

### Existing variables whose names were broadened

- `LLM_MODEL` continues to serve as the Groq model name; the field is
  also aliased as `GROQ_MODEL` for legacy Chat Safety readers.
- `LLM_TIMEOUT` is aliased as `GROQ_TIMEOUT_SECONDS`.

### Values with non-default status

- `GROQ_API_KEY` — required, no default.
- `GUARDRAIL_API_TOKEN` — required, no default.
- `LLM_CONSENSUS_ATTEMPTS` — default changed to `1`.
- `LLM_FALLBACK_ENABLED` — default `true`.

No secrets are committed. `.env.example` documents every variable.

---

## 9. Remaining limitations and open decisions

### Deliberately deferred

- **`Guardrail` ABC** — not introduced. The `GuardrailResult` data
  shape exists but is not wrapped in an interface. When a second
  capability needs it, promote both to `app/guardrails/contracts.py`.
- **LangGraph integration for Chat Safety** — the guardrail is called
  directly. Graph orchestration is not needed for a single classification.
- **Async conversion of the Chat Safety chain** — the sync path is
  preserved via `LLMGateway.generate_sync()`. A future refactor could
  make the whole chain async if needed.
- **Route-level tests for `test_api.py`, `test_metrics.py`,
  `test_response_fields.py`** — moved into
  `tests/integration/chat_safety/` in step 10.

### Production release gates (unchanged from plan §7.2)

These are **not** integration items. They require explicit sign-off
before production traffic:

- [ ] Fail-open policy approved in writing by the Swabi product/backend
      owner.
- [ ] Token rotation procedure documented and rehearsed once on staging.
- [ ] Deployment worker count decided and documented (metrics and
      circuit state are per-process).
- [ ] Rate-limit expectation confirmed with the NestJS team.
- [ ] Manager sign-off on the failure policy and the auth boundary.

### Known operational characteristics

- `/metrics` and `/health` are unauthenticated. Bind behind a reverse
  proxy that restricts inbound access.
- Counters and circuit state are in-memory per process. Multi-worker
  deployments show partial values unless aggregation is added.
- The Gemini fallback is configured but has not been exercised against
  the live API in CI. Mocked tests cover the dispatch path.

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

**The integration phase is complete and ready for review.** It is not
production-ready until the release gates in §9 are closed.