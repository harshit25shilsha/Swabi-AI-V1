# Swabi AI Service — Architecture

**Status:** Post Chat Safety integration + post-integration hardening +
Phase 1 (Enquiry Understanding)
**Last updated:** October 2026
**Audience:** Engineers working on the Swabi AI service or integrating a
new AI capability.

This document describes what the service is, how the pieces fit together,
and the conventions a new capability should follow. It is not a
specification of the Swabi marketplace (see the BRD) and not a
tutorial for any specific AI technique.

---

## 1. What this service is

A standalone Python / FastAPI service that provides AI capabilities to
the Swabi marketplace. It is called **by** the Swabi NestJS backend
(and, for user-facing routes, by the frontend through NestJS). It is
never called directly by a browser for authenticated actions, and it
never talks to the Swabi PostgreSQL database.

```
Frontend  ──▶  NestJS  ──▶  AI Service  ──▶  LLM providers
                  │             │
                  │             └──▶  NestJS APIs (for live data)
                  │
                  └──▶  PostgreSQL / Redis / business services
```

**Rule:** NestJS is the source of truth for business state and
authorization. The AI service provides understanding, reasoning,
retrieval, and recommendation.

---

## 2. Layout

```
app/
├── main.py                     FastAPI app: routes, middleware, exception handlers
├── core/                       Config, logging, exceptions, Redis,
│                               request-scoped auth context, service auth
├── llm/                        Shared LLM gateway + provider clients
├── capabilities/               Feature-specific code
│   ├── chat_safety/            Chat moderation
│   │   ├── detectors/          Deterministic rules (phone, email, URL,
│   │   │                       scattered numbers, disguised runs, etc.)
│   │   ├── policies/           Evidence → decision mapping
│   │   ├── prompts.py          SYSTEM_PROMPT + PROMPT_VERSION
│   │   ├── schemas.py          Request/response + internal result types
│   │   ├── llm_adapter.py      Adapter over the shared gateway
│   │   ├── moderator.py        Provider ladder + circuit breakers + consensus
│   │   ├── service.py          Orchestrator (deterministic → LLM → policy)
│   │   └── dependencies.py     FastAPI dependencies (auth, re-export)
│   └── enquiry/                Travel request → structured enquiry fields
│       ├── constants.py        Enums, field names, thresholds
│       ├── schemas.py          Extraction + request/response
│       ├── prompts.py          Extraction prompt + version hash
│       └── service.py          Extraction + missing-field detection
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 FastAPI routers
│                               (health, metrics, chat_safety, enquiry)
├── swabi/                      Async client for NestJS APIs (no direct DB)
├── rag/                        Retriever interface (Phase 2 will implement)
├── graph/                      LangGraph foundation (minimal)
└── evaluation/                 Dataset loader + runner

tests/
├── unit/                       Pure logic, mocks, no external calls
├── integration/                FastAPI + mocks; live providers optional
└── contract/                   Public API contract guards

docs/
├── architecture.md             This file
├── guardrail_integration_plan.md
├── gdv1_test_verification.md
├── phase_guardrail_integration_report.md
└── phase_enquiry_report.md
```

---

## 3. Routes

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Liveness. Returns status, uptime, versions. |
| `GET` | `/ready` | none | Readiness. Checks Redis. |
| `GET` | `/metrics` | none | Prometheus text. Counters + circuit state. |
| `POST` | `/api/v1/validate-message` | service token | Chat Safety classification. |
| `POST` | `/api/v1/enquiry/understand` | service token | Enquiry Understanding extraction. |

Public/unauthenticated routes must sit behind a reverse proxy that
restricts inbound access. This is documented and expected.

---

## 4. Middleware

Middleware is registered in `app/main.py`. FastAPI/Starlette wraps them
LIFO: the **last one added is the outermost** and runs first on the
request, last on the response.

Current order:

```
request_id → auth_context → observability → route
```

Registered in reverse (observability, auth_context, request_id).

| Middleware | Responsibility |
|---|---|
| `request_id_middleware` | Generate or accept `X-Request-ID`, set `request_id_var`, add header to response. |
| `auth_context_middleware` | Populate `user_jwt_var` **only** for allow-listed user-facing routes; clear it otherwise. |
| `ObservabilityMiddleware` | Time the request, emit the per-request moderation log line, update counters. |

The order is verified by `tests/integration/test_middleware_order.py`.
Do not change registration order without updating those tests.

---

## 5. Authentication model

There are **two distinct auth flows**. They must not be conflated.

### 5.1 Inbound — service-to-service

Both `/api/v1/validate-message` and `/api/v1/enquiry/understand` are
called by the Swabi NestJS backend with a shared service credential:

```
Authorization: Bearer <GUARDRAIL_API_TOKEN>
```

The check lives in `app/core/service_auth.py::verify_service_token`
(`secrets.compare_digest`). Chat Safety's `dependencies.py` re-exports
it for backward compatibility with existing imports. Invalid tokens →
401.

This token is **not** a user JWT. It carries no user identity, no role,
no permissions. Treat it as a shared secret between two backend
services.

The name `GUARDRAIL_API_TOKEN` is historical — it was introduced when
Chat Safety was the only service-to-service route. It is now the general
inbound credential for the AI service. Renaming it to `AI_SERVICE_TOKEN`
is planned but requires coordination with the NestJS team, which sends
this value. Do not rename it unilaterally.

### 5.2 Outbound — user JWT forwarding

User-facing AI routes (added in later phases) accept the logged-in
user's JWT and forward it unchanged to NestJS when calling
`SwabiClient`. NestJS authorizes against the user's real identity.

`SwabiClient._headers()` prefers the caller's JWT if present, falling
back to a service API key for internal calls.

### 5.3 The boundary

`auth_context_middleware` only captures the user JWT for routes
under `USER_JWT_ROUTE_PREFIXES`. Service-to-service routes — including
`/api/v1/validate-message` and `/api/v1/enquiry/understand` — are
excluded and explicitly clear the context. This prevents a service
token from leaking into `SwabiClient` as if it were a user JWT.

Do not add a route to the allow-list unless it genuinely acts on behalf
of a logged-in user. The default is "no capture".

---

## 6. LLM access

All LLM calls go through the shared `LLMGateway` in `app/llm/`. No
capability should instantiate a provider SDK directly.

```
LLMGateway
    ├── GroqProvider (async + sync via httpx)
    ├── generate_structured()        (async, Pydantic-validated JSON)
    └── generate_structured_sync()   (sync, Pydantic-validated JSON)
```

`GroqProvider` implements bounded retries, timeouts, and normalized
errors. `generate_structured()` / `generate_structured_sync()` ask the
model for JSON, validate against a Pydantic schema, and retry once on
failure.

Both current capabilities use the **sync** path because their service
chains are synchronous:

- Chat Safety uses `generate_sync()` through `chat_safety/llm_adapter.py`.
- Enquiry Understanding uses `generate_structured_sync()` directly.

New capabilities should prefer the async interface if they can be async
end-to-end. The sync path exists because rewriting both capabilities to
async is a separate, larger change (see §13).

### Chat Safety's provider ladder

Chat Safety additionally uses a Gemini fallback via `google-genai`. The
ladder is:

```
Groq → Gemini
```

Each provider has its own circuit breaker. If Groq fails or its circuit
is open, Gemini is tried. If both fail or are open, the guardrail fails
open unless a soft detector already flagged REVIEW (see §10).

Enquiry Understanding does **not** use the Gemini fallback. A failed
extraction has no safe fallback, so it fails closed (502/504) and the
frontend prompts a retry. See §14.

The fail-open policy is a **production release gate** — see
`docs/guardrail_integration_plan.md` §7.2. Do not change it without
sign-off.

### Connection pooling

`GroqProvider.generate_sync()` uses a module-level shared `httpx.Client`
with `max_keepalive_connections=20`, `max_connections=50`, and
`keepalive_expiry=30s`. The client is created lazily on first use and
closed on FastAPI shutdown (and again via `atexit` as a safety net).

The async `generate()` path still constructs a fresh `AsyncClient` per
attempt and therefore does not pool. This is intentional while both
capabilities use the sync path. The async refactor should add a shared
`AsyncClient` — see §13.

### Retry policy

Initial delay is 0.2s with exponential backoff (`0.2`, `0.4`, `0.8`).
With `LLM_MAX_RETRIES=2`, worst-case wait before fail-open is ~0.6s plus
the last call's own latency. `generate_sync` passes `timeout=self.timeout`
to the shared client per request, so a provider-level timeout override is
honored even though the client has its own default.

### Reasoning models

`openai/gpt-oss-120b` is a reasoning model: it spends completion tokens
on internal reasoning *before* emitting the JSON payload. `LLM_MAX_TOKENS`
must leave room for both phases. 128 was tried and rejected the request
with `json_validate_failed` and empty output. 512 is the current floor.

Both `generate()` and `generate_sync()` retry once without
`response_format={"type":"json_object"}` when Groq returns 400 with
`json_validate_failed`. The prompt already asks for JSON, and the
moderator's `_parse_llm_json` handles fenced output.

---

## 7. Adding a new capability

The pattern established by `chat_safety/` and refined by `enquiry/`:

1. Create `app/capabilities/<name>/`.
2. Put capability-specific code in that folder. Do not reach into
   `chat_safety/` or `enquiry/`.
3. Define the request/response Pydantic models in `schemas.py`.
4. If the capability needs to call an LLM, call `LLMGateway`, not a
   provider SDK.
5. If it needs to call NestJS, use `SwabiClient`. Do not open a raw
   `httpx` connection.
6. Add a thin route in `app/api/routes/`. The route should validate,
   authenticate, invoke the service, and serialize.
7. Add tests under `tests/unit/<name>/` and `tests/integration/<name>/`.
8. If the route is service-to-service (not user-facing), do not add it
   to `USER_JWT_ROUTE_PREFIXES`.
9. If the route is user-facing, add its prefix to the allow-list and
   add a test asserting `user_jwt_var` is populated.

### Two capability templates

| Template | When to use | Files |
|---|---|---|
| **Stateful classification** (Chat Safety) | Message-in / decision-out, with deterministic pre-filters | `detectors/`, `policies/`, `prompts.py`, `schemas.py`, `llm_adapter.py`, `moderator.py`, `service.py`, `dependencies.py` |
| **Pure extraction** (Enquiry Understanding) | Free-text-in / structured-data-out | `constants.py`, `schemas.py`, `prompts.py`, `service.py` |

Enquiry Understanding is the minimum viable capability: four files, no
detectors, no policy layer, no LLM adapter. It calls
`LLMGateway.generate_structured_sync()` directly. Use it as the template
for any future capability that is a pure text → structured-data
transform.

### Guardrails

`GuardrailResult` and `GuardrailDecision` are defined in
`app/capabilities/chat_safety/schemas.py`. A future capability that
produces a guardrail-style decision may reuse this shape. When a second
capability needs it, promote both to `app/guardrails/contracts.py`. Do
not create that module preemptively.

Enquiry Understanding has a **different** result shape
(`EnquiryUnderstandResponse`) because it is not a guardrail. Do not
force-fit its response into `GuardrailResult`.

Do not build a plugin registry, dynamic discovery system, or ABC for
capabilities. Explicit imports are clearer.

---

## 8. Configuration

All configuration lives in `app/core/config.py::Settings`. Field names
are lower-case Pydantic names. Upper-case properties exist as read-only
aliases for legacy code compatibility (mostly Chat Safety's internal
readers).

**Required with no default:**

- `GROQ_API_KEY`
- `GUARDRAIL_API_TOKEN`

**Have safe development defaults:** everything else.

Never commit `.env`. The template is `.env.example`.

`LLM_CONSENSUS_ATTEMPTS` defaults to `1` (a single LLM call per message,
except when the provider ladder itself retries). This is deliberate: see
integration plan §5.2. Raising it requires evaluation evidence.

`LLM_MAX_TOKENS` defaults to `512`. Do not reduce it below ~256 for
reasoning models; see §6.

Both current capabilities reuse the same settings. There are no
capability-specific environment variables today. If a future capability
needs one, prefer a lower-case Pydantic field with a sane default over
a new required variable.

---

## 9. Observability

- **Request correlation:** every request gets an `X-Request-ID`. If the
  caller supplies one, it is preserved. Otherwise a `gr_<16-hex>` is
  generated.
- **Structured logs:** JSON by default; each record includes `request_id`
  and any `extra={...}` fields. Sensitive fields (JWTs, service tokens,
  raw messages) are never logged.
- **Message hashing:** the Chat Safety route logs a truncated SHA-256 of
  the message, never the message itself.
- **Enquiry event:** the Enquiry Understanding service emits one
  `enquiry_understood` event per request, carrying `missing_fields`,
  `questions_count`, `completeness`, and `prompt_version`. The raw
  message is not logged.
- **Counters:** in-memory, thread-safe, exposed via `/metrics`. Reset on
  process restart; not shared across workers. Today the counters are
  Chat Safety-specific (`guardrail_*`); Enquiry does not increment them.
- **Circuit breakers:** per provider, Chat Safety only. State is exposed
  via `/metrics`.

For single-worker deployments, `/metrics` reflects the whole process.
Multi-worker deployments will show partial values per worker. Choose
worker count and metric aggregation deliberately.

---

## 10. Failure behavior

### Chat Safety

| Failure | Behavior |
|---|---|
| Groq timeout / 5xx | Retried (bounded); then the ladder falls over to Gemini. |
| Groq returns 400 `json_validate_failed` | Retried once without `response_format`; if that fails, ladder falls to Gemini. |
| Gemini unavailable | Circuit opens; the ladder exhausts. |
| All circuits open | `CircuitOpenError` raised. Service fails open unless soft REVIEW. |
| Invalid JSON from LLM | Moderator attempts to salvage; on failure, fails open. |
| Detector raises | Logged; other detectors continue. |
| Soft detector says REVIEW, LLM says ALLOW | Response escalates to `source="review"` with `allowed=false, action=BLOCK`. |
| Redis unavailable | `/ready` reports degraded; the service keeps serving. |
| Invalid auth token | 401 immediately. No retry. |

**Fail-open on provider outage is a deliberate product decision.** It is
preserved from the original Guardrail Service. Changing it requires
sign-off from the Swabi product owner.

### Enquiry Understanding

| Failure | Behavior |
|---|---|
| Groq timeout | Retried (bounded); then returns 504. |
| Groq 5xx | Retried (bounded); then returns 502. |
| Groq returns 400 `json_validate_failed` | Retried once without `response_format`. |
| Invalid JSON from LLM | `generate_structured_sync` retries once; then 422. |
| Invalid request body | 422 immediately (Pydantic). |
| Invalid auth token | 401 immediately. |
| All providers unavailable | 502. |

**Enquiry Understanding fails closed.** There is no fail-open policy.
A failed extraction has no safe fallback — the frontend shows a retry
prompt or the user enters the enquiry manually. This is materially
different from Chat Safety and is deliberate: extraction errors are
user-visible and correctable; message-delivery errors are not.

---

## 11. Testing conventions

- **Unit tests** (`tests/unit/`): pure logic, mocked dependencies. No
  network, no live providers.
- **Integration tests** (`tests/integration/`): FastAPI `TestClient` +
  mocks. Same rules as unit tests for the default run.
- **Contract tests** (`tests/contract/`): guard the public API shape.
  A field added or removed in `ModerationResponse` or
  `EnquiryUnderstandResponse` must break a test.
- **Live integration tests** (marked `@pytest.mark.integration`): hit
  real Groq / Gemini. Opt-in only. Default `pytest` run excludes them
  via `addopts = "-m 'not integration'"`.

Run:

```bash
pytest                          # unit + integration, mocked
pytest -m integration -v        # live providers (requires real keys)
ruff check .                    # lint
ruff format --check .           # formatting
```

---

## 12. What not to do

- Do not import from one capability into another
  (`app.capabilities.chat_safety.*` or `app.capabilities.enquiry.*`).
  If code is genuinely shared, move it to `app/core/`, `app/llm/`, or
  `app/observability/`.
- Do not access the Swabi PostgreSQL database. Use `SwabiClient`.
- Do not instantiate a provider SDK outside `app/llm/` (and the Gemini
  branch of `chat_safety/llm_adapter.py`, which is the one sanctioned
  exception).
- Do not add a route to `USER_JWT_ROUTE_PREFIXES` unless it acts on
  behalf of a user.
- Do not add new configuration for providers that are not in use.
- Do not create a `Guardrail` ABC or registry until a second
  capability needs it.
- Do not change the Chat Safety fail-open policy without product
  sign-off.
- Do not log the raw message, JWTs, or service tokens.
- Do not reduce `LLM_MAX_TOKENS` below ~256 for reasoning models.
- Do not reflow the string body of any `prompts.py` file. The prompt
  bytes are hashed into a version identifier and reflowing changes the
  hash.

---

## 13. Open items

Referenced here so they are not lost:

- **Chat Safety fail-open policy sign-off** — see integration plan §7.2.
- **Token rotation procedure** — see the Chat Safety README section.
- **Deployment worker count decision** — metrics are per-process.
- **Rate-limit expectation with NestJS** — who throttles, at what point.
- **Async refactor of the LLM path** — currently only `generate_sync()`
  pools connections; `generate()` opens a fresh `AsyncClient` per
  attempt. Both live capabilities are sync today, so the cost is zero.
  Do the refactor when a capability needs to be async end-to-end, or
  when thread-pool saturation shows in metrics.
- **Tail latency policy** — P95 on the sync path can reach 4s under
  provider stress. Currently bounded by `LLM_TIMEOUT` (default 60s) and
  `LLM_MAX_RETRIES` (default 2). Decide a client-visible deadline before
  production.
- **Enquiry Tier 2 personalization** — deferred. Would read customer
  profile and family members from NestJS to pre-fill known ages and
  currency. Not needed for the current form, which collects travellers
  as a single integer. Add only if the UI exposes the split.
- **`GUARDRAIL_API_TOKEN` rename** — historical name; the value is now
  the general AI service token. Rename to `AI_SERVICE_TOKEN` requires
  coordination with NestJS. Not urgent.

---

## 14. Enquiry Understanding

### What it does

`POST /api/v1/enquiry/understand` receives a free-text travel request
and returns:

1. A structured `EnquiryExtraction` object (20 optional fields).
2. A list of missing field names.
3. Clarifying questions for the most important missing fields.
4. A completeness score (0.0–1.0) for the frontend to display.

Read-only. It does not create enquiries, call vendors, or touch any
persistent state. It is a pure text → JSON transform.

### Why the shape is a superset

The extraction schema covers three sources:

- the **live Enquiry form** (13 fields — the source of truth for what
  the UI renders)
- the **`CreateEnquiryDto`** in the NestJS Swagger contract
- the **Swabi BRD** `CUST-ENQ-*` fields (for future UI additions)

When the UI and BRD disagree, both fields are included. Fields the UI
does not render today (`rooms`, `property_type`, `trip_type`, `flags`,
`adults`, `children[]`) are populated when the user mentions them and
dormant otherwise. This avoids a second extraction pass when the two
specs converge.

Where the DTO and UI disagree (e.g. accommodation is single-select in
the DTO, multi-select in the UI), the AI follows the **DTO** so the
frontend has no translation layer. If a future DTO supports arrays,
that is a one-line change in `constants.py` and `prompts.py`.

### Call path

Service-to-service. NestJS calls the route with the shared service
token. It is **not** added to `USER_JWT_ROUTE_PREFIXES` — the AI service
does not receive or forward user JWTs on this route.

### Determinism

The extraction itself is LLM-driven and will vary slightly between runs.
Everything downstream is deterministic:

- `missing_fields` — computed from the extraction in a fixed order.
- `clarifying_questions` — templates keyed by field name, capped at 4.
- `completeness` — `1.0 - (required_missing / required_total)`.

The frontend can rely on these being stable for a given extraction.

### Date resolution

The prompt resolves:

- **Explicit dates** (`"15/12/26"`) → `dd/mm/yy`
- **Ordinal weekdays** (`"first Saturday of December"`, `"next Monday"`)
  → resolved using the current date
- **Ranges** → `travel_date` + `travel_date_end`
- **Duration-derived end** → `travel_date_end = travel_date + duration_days - 1`

**Month-only expressions** (`"in December"`) leave `travel_date` null.
The date picker on the frontend collects the actual date. The prompt
explicitly forbids inventing a specific date from a month-only phrase.

### Enum values

Every enum string the frontend form uses is embedded in the prompt and
in `constants.py`. If the frontend changes a value, both must be updated
and `ENQUIRY_PROMPT_VERSION` will change automatically.

See the README "Enquiry Understanding → Enum values" section for the
full list.

### Failure behavior

Enquiry Understanding fails closed, unlike Chat Safety. A failed
extraction returns 502 or 504 and the frontend prompts a retry. There is
no fail-open path because a partial or missing extraction is worse than
an explicit error.

### Extending

To add a field:

1. Add the field to `EnquiryExtraction` in `schemas.py`.
2. Add it to the field list and (if needed) an enum in `constants.py`.
3. Add it to `REQUIRED_FIELDS` or `ASKABLE_FIELDS` in `constants.py`.
4. Update the prompt in `prompts.py`. `ENQUIRY_PROMPT_VERSION` will
   change automatically.
5. Update the contract test's field-set assertions.
6. Add or update a unit test for missing-field detection.

The extractor is deliberately one prompt and one schema. Do not split
it into multiple LLM calls unless accuracy demands it.