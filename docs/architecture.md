# Swabi AI Service — Architecture

**Status:** Post Chat Safety integration
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
├── core/                       Config, logging, exceptions, Redis, request-scoped auth context
├── llm/                        Shared LLM gateway + provider clients
├── capabilities/               Feature-specific code
│   └── chat_safety/            First capability: chat moderation
│       ├── detectors/          Deterministic rules (phone, email, URL, etc.)
│       ├── policies/           Evidence → decision mapping
│       ├── prompts.py          SYSTEM_PROMPT + PROMPT_VERSION
│       ├── schemas.py          Request/response + internal result types
│       ├── llm_adapter.py      Adapter over the shared gateway
│       ├── moderator.py        Provider ladder + circuit breakers + consensus
│       ├── service.py          Orchestrator (deterministic → LLM → policy)
│       └── dependencies.py     FastAPI dependencies (auth)
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 FastAPI routers (health, metrics, chat_safety)
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
└── phase_guardrail_integration_report.md
```

---

## 3. Routes

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Liveness. Returns status, uptime, versions. |
| `GET` | `/ready` | none | Readiness. Checks Redis. |
| `GET` | `/metrics` | none | Prometheus text. Counters + circuit state. |
| `POST` | `/api/v1/validate-message` | service token | Chat Safety classification. |

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

The Swabi NestJS backend calls `/api/v1/validate-message` with a shared
service credential:

```
Authorization: Bearer <GUARDRAIL_API_TOKEN>
```

`verify_token` in `app/capabilities/chat_safety/dependencies.py` checks
the token using `secrets.compare_digest`. Invalid tokens → 401.

This token is **not** a user JWT. It carries no user identity, no role,
no permissions. Treat it as a shared secret between two backend
services.

### 5.2 Outbound — user JWT forwarding

User-facing AI routes (e.g. enquiry understanding in a later phase)
accept the logged-in user's JWT and forward it unchanged to NestJS when
calling `SwabiClient`. NestJS authorizes against the user's real identity.

`SwabiClient._headers()` prefers the caller's JWT if present, falling
back to a service API key for internal calls.

### 5.3 The boundary

`auth_context_middleware` only captures the user JWT for routes
under `USER_JWT_ROUTE_PREFIXES`. Service-to-service routes — including
`/api/v1/validate-message` — are excluded and explicitly clear the
context. This prevents the guardrail's service token from leaking into
`SwabiClient` as if it were a user JWT.

Do not add a route to the allow-list unless it genuinely acts on behalf
of a logged-in user. The default is "no capture".

---

## 6. LLM access

All LLM calls go through the shared `LLMGateway` in `app/llm/`. No
capability should instantiate a provider SDK directly.

```
LLMGateway
    ├── GroqProvider (async + sync via httpx)
    └── generate_structured()  (Pydantic-validated JSON)
```

`GroqProvider` implements bounded retries, timeouts, and normalized
errors. `LLMGateway.generate_structured()` asks the model for JSON,
validates against a Pydantic schema, and retries once on failure.

The Chat Safety capability uses `generate_sync()` (the synchronous
mirror) because its service chain is synchronous. This is the only
sync path; new capabilities should prefer the async interface.

### Chat Safety's provider ladder

Chat Safety additionally uses a Gemini fallback via `google-genai`. The
ladder is:

```
Groq → Gemini
```

Each provider has its own circuit breaker. If Groq fails or its circuit
is open, Gemini is tried. If both fail or are open, the guardrail fails
open (returns `ALLOW` with `source="fail_open"`).

The fail-open policy is a **production release gate** — see
`docs/guardrail_integration_plan.md` §7.2. Do not change it without
sign-off.

---

## 7. Adding a new capability

The pattern established by `chat_safety/`:

1. Create `app/capabilities/<name>/`.
2. Put detectors, policy, prompts, and schemas in that folder. Do not
   reach into `chat_safety/`.
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

### Guardrails

`GuardrailResult` and `GuardrailDecision` are defined in
`app/capabilities/chat_safety/schemas.py`. A future capability that
produces a guardrail-style decision may reuse this shape. When a second
capability needs it, promote both to `app/guardrails/contracts.py`. Do
not create that module preemptively.

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
- **Counters:** in-memory, thread-safe, exposed via `/metrics`. Reset on
  process restart; not shared across workers.
- **Circuit breakers:** per provider. State is exposed via `/metrics`.

For single-worker deployments, `/metrics` reflects the whole process.
Multi-worker deployments will show partial values per worker. Choose
worker count and metric aggregation deliberately.

---

## 10. Failure behavior

| Failure | Behavior |
|---|---|
| Groq timeout / 5xx | Retried (bounded); then the ladder falls over to Gemini. |
| Gemini unavailable | Circuit opens; the ladder exhausts. |
| All circuits open | `CircuitOpenError` raised. Service fails open. |
| Invalid JSON from LLM | Moderator attempts to salvage; on failure, fails open. |
| Detector raises | Logged; other detectors continue. |
| Redis unavailable | `/ready` reports degraded; the service keeps serving. |
| Invalid auth token | 401 immediately. No retry. |

**Fail-open on provider outage is a deliberate product decision.** It is
preserved from the original Guardrail Service. Changing it requires
sign-off from the Swabi product owner.

---

## 11. Testing conventions

- **Unit tests** (`tests/unit/`): pure logic, mocked dependencies. No
  network, no live providers.
- **Integration tests** (`tests/integration/`): FastAPI `TestClient` +
  mocks. Same rules as unit tests for the default run.
- **Contract tests** (`tests/contract/`): guard the public API shape.
  A field added or removed in `ModerationResponse` must break a test.
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

- Do not import from `app.capabilities.chat_safety.*` in another
  capability. If code is genuinely shared, move it to `app/core/`,
  `app/llm/`, or `app/observability/`.
- Do not access the Swabi PostgreSQL database. Use `SwabiClient`.
- Do not instantiate a provider SDK outside `app/llm/` (and the Gemini
  branch of `chat_safety/llm_adapter.py`, which is the one sanctioned
  exception).
- Do not add a route to `USER_JWT_ROUTE_PREFIXES` unless it acts on
  behalf of a user.
- Do not add new configuration for providers that are not in use.
- Do not create a `Guardrail` ABC or registry until a second
  capability needs it.
- Do not change the fail-open policy without product sign-off.
- Do not log the raw message, JWTs, or service tokens.

---

## 13. Open items

Referenced here so they are not lost:

- Fail-open policy sign-off — see integration plan §7.2.
- Token rotation procedure — see the Chat Safety README section.
- Deployment worker count decision — metrics are per-process.
- Rate-limit expectation with NestJS — who throttles, at what point.
- `docs/phase_guardrail_integration_report.md` — final report from the
  Chat Safety integration.