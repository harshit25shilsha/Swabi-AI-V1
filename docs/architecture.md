# Swabi AI Service — Architecture

**Status:** Chat Safety + Phase 1 (Enquiry Understanding) + Phase 2 AI
Support/RAG feature scope integrated; broader production rollout checks remain
**Last updated:** November 2026
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
Frontend ──▶ NestJS ──▶ AI Service ──▶ Groq / Gemini
               │             ├──▶ Qdrant (approved knowledge)
               │             └──▶ NestJS APIs (user-authorized live reads)
               └──▶ PostgreSQL / Redis / business services
```

**Rule:** NestJS is the source of truth for business state and
authorization. The AI service provides moderation, extraction, support
retrieval, and answer generation; it does not perform business writes.

---

## 2. Layout

```
app/
├── main.py                     FastAPI app: routes, middleware, exception handlers
├── core/                       Config, logging, exceptions, Redis,
│                               response cache, request-scoped auth context,
│                               service auth
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
│   └── support/                Audience-aware RAG and live Swabi data
│       ├── intent_resolver.py  Deterministic + LLM intent routing
│       ├── live_data.py        Allowlisted NestJS reads, auth, decryption
│       ├── live_data_format.py Safe, user-facing data formatting
│       ├── retrieval_query.py  Canonical retrieval query preparation
│       └── service.py          RAG/live orchestration and response checks
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 FastAPI routers
│                               (health, metrics, chat_safety, enquiry, support)
├── swabi/                      NestJS transport and encrypted-response helpers
├── rag/                        Lifecycle-aware ingestion, Qdrant, hybrid RAG
├── graph/                      LangGraph foundation (minimal)
└── evaluation/                 Retrieval and grounding evaluation

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
| `POST` | `/api/v1/validate-message` | gateway assertion + `ai:chat-safety` | Chat Safety classification. |
| `POST` | `/api/v1/enquiry/understand` | gateway assertion + `ai:enquiry:understand` | Enquiry Understanding extraction. |
| `POST` | `/api/v1/support/ask` | gateway assertion + `ai:support:ask` | Knowledge-grounded support and delegated live reads. |

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

### Target state: NestJS gateway verification

NestJS is the public authentication gateway for every AI route. It validates
the user JWT, derives the authenticated user, audience, and authorized AI
scopes, then calls the private AI service with the original Bearer JWT plus a
signed, short-lived principal assertion. The AI service verifies that
assertion before it performs any LLM, RAG, or live-data work.

`app/core/gateway_auth.py` validates the assertion HMAC and expiry;
`app/core/route_auth.py` requires the route capability. Missing, invalid, or
expired authentication context returns `401`; an authenticated principal that
lacks the capability returns `403`.

The AI service treats the user JWT as opaque. For user-specific live reads it
forwards that same token to NestJS, where NestJS remains authoritative for the
particular bookings, wallet, payments, enquiries, or vendor resource. No
generic service token may replace a user JWT for a personal-data request.

The full header format and deployment requirements are in
[`auth_gateway_contract.md`](auth_gateway_contract.md). AI must be private;
NestJS/public ingress must strip the trusted-principal headers from browser
traffic and use mTLS or equivalent workload authentication to AI.

### Legacy migration mode (temporary)

`AI_AUTH_MODE=legacy_service_token` preserves the prior behavior while NestJS
is being upgraded. It uses `GUARDRAIL_API_TOKEN` for Chat Safety and Enquiry.
Support still forwards an opaque user token for live reads, but its KB-only
path only checks for a non-empty Bearer token. This mode therefore does not
close the KB authentication gap and is not the production target.

### Legacy behavior retained during migration

The remaining subsections describe the old behavior retained only when
`AI_AUTH_MODE=legacy_service_token`; they are not the gateway-mode design.

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

`POST /api/v1/support/ask` requires a user bearer token. The route extracts
it and passes it explicitly to the support service. Private live-data
fetchers forward it unchanged to NestJS; NestJS authorizes against the user's
real identity. Public catalogue intents may call their endpoints without a
JWT.

The support integration currently uses its own `httpx` transport in
`app/capabilities/support/live_data.py`, including response-envelope
unwrapping and optional encrypted-payload decryption. Do not describe this
path as using `SwabiClient` until it is actually consolidated there.

### 5.3 The boundary

`auth_context_middleware` only captures the user JWT for routes
under `USER_JWT_ROUTE_PREFIXES`. Service-to-service routes — including
`/api/v1/validate-message` and `/api/v1/enquiry/understand` — are
excluded and explicitly clear the context. The support route currently
validates its own bearer header and passes the token explicitly; it does not
rely on the auth-context allow-list. This prevents a service token from
being treated as a user JWT.

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

`GroqProvider` implements bounded retries, timeouts, token pacing, and
normalized errors. `generate_structured()` / `generate_structured_sync()`
ask the model for JSON and validate it against a Pydantic schema. If the
provider marks a response as completion-truncated, the gateway retries once
with a larger completion budget (up to 4,096 tokens); other validation
failures use the normal single retry.

Current capability call paths:

- Chat Safety uses `generate_sync()` through `chat_safety/llm_adapter.py`.
- Enquiry Understanding uses `generate_structured_sync()` directly.
- RAG support uses structured synchronous generation for knowledge answers.
  Successful live API answers are formatted deterministically and do not call
  the LLM.

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
attempt and therefore does not pool. The sync path is used by current
capability orchestration; add a shared `AsyncClient` if an end-to-end async
capability makes pooling necessary.

### Retry policy

Initial delay is 0.2s with exponential backoff (`0.2`, `0.4`, `0.8`).
With `LLM_MAX_RETRIES=2`, worst-case wait before fail-open is ~0.6s plus
the last call's own latency. `generate_sync` passes `timeout=self.timeout`
to the shared client per request, so a provider-level timeout override is
honored even though the client has its own default.

### Reasoning models

`openai/gpt-oss-120b` is a reasoning model: it spends completion tokens
on internal reasoning *before* emitting the JSON payload. `LLM_MAX_TOKENS`
must leave room for both phases. General `LLM_MAX_TOKENS` defaults to 512;
RAG answers use `SUPPORT_ANSWER_MAX_TOKENS` (1,536 by default). The gateway
observes the provider `finish_reason` and increases the budget on a
truncation retry.

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

Gateway mode additionally requires `AI_GATEWAY_AUTH_SECRET`. It must be the
same secret in the NestJS and AI-service environments, is used only to sign
and verify the private principal assertion, and is not a JWT signing key.
`AI_AUTH_MODE` defaults to `gateway`; use `legacy_service_token` only during
the staged NestJS migration.

**Have safe development defaults:** everything else.

Never commit `.env`. The template is `.env.example`.

`LLM_CONSENSUS_ATTEMPTS` defaults to `1` (a single LLM call per message,
except when the provider ladder itself retries). This is deliberate: see
integration plan §5.2. Raising it requires evaluation evidence.

`LLM_MAX_TOKENS` defaults to `512`. Do not reduce it below ~256 for
reasoning models; RAG answers have a separate `SUPPORT_ANSWER_MAX_TOKENS`
budget.

AI Support settings:

| Variable | Default | Purpose |
|---|---:|---|
| `SUPPORT_MAX_CONTEXT_CHUNKS` | `5` | Maximum retrieved chunks sent to answer generation. |
| `SUPPORT_MAX_CHUNK_CHARS` | `1200` | Per-chunk text limit in the support context. |
| `SUPPORT_ANSWER_MAX_TOKENS` | `1536` | Structured RAG answer completion budget (bounded to 4,096). |
| `RAG_RERANKER_ENABLED` | `false` | Whether to apply the optional cross-encoder after RRF. Keep disabled based on the latest benchmark. |
| `RAG_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Local reranker model. |

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
- **AI Support:** logs a question hash, audience, intent/source, answer type,
  citation count, retrieval result count, and live-data source. Raw JWTs and
  question text are not included.
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
- Do not access the Swabi PostgreSQL database. Use the approved NestJS API
  integration (`SwabiClient` or the support live-data transport).
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
  attempt. Current capability orchestration uses synchronous call paths.
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
- **Phase 2 grounding acceptance** — rerun the semantic grounding evaluation
  after the completion-budget and citation-context changes, review per-case
  findings, and agree on product-owned quality thresholds.
- **Phase 2 end-to-end acceptance** — verify real NestJS responses, auth,
  decryption, empty results, error handling, and stale-policy rejection
  during an index promotion. The automated suite does not prove deployment
  integration.
- **Package search constraints** — the current NestJS package API supports
  `page`, `limit`, and `destination` only. Season, date availability, budget,
  and group-size filtering require backend API support.
- **Phase 3: Bid and Offer Analyzer** — not started; begin after Phase 2
  acceptance is complete.

---

## 14. Enquiry Understanding

### What it does

`POST /api/v1/enquiry/understand` receives a free-text travel request
and returns:

1. A structured `EnquiryExtraction` object (21 optional fields).
2. A list of genuinely unanswered required field names.
3. Exactly one clarifying question for each missing required field.
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
does not render today (`rooms`, `property_type`, `property_preferences`, `trip_type`, `flags`,
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
- `clarifying_questions` — one template-backed question for every missing
  required field, in the same order as `missing_fields`.
- `completeness` — `completed_required_fields / 9`, rounded to four decimal
  places.

The frontend can rely on these being stable for a given extraction.

### Required-answer model

The required fields are, in order: `destination`, `travel_date`,
`travel_date_end`, `accommodation`, `transport`, `meals`, `tour_guide`,
`activities`, and `special_requests`. Budget, party size, duration, room
count, date flexibility, and other descriptive fields are useful extraction
data but do not affect completeness.

`property_preferences` is an optional list for explicitly stated Travel for
Work, Travel with Pets, Entire Apartment, and Couple Friendly preferences. It
is returned when present but never creates a missing field or question.

`null` or an empty list means unknown. Explicitly declined services are
preserved as values such as `"No transport required"`,
`["No Meals Required"]`, `"No"` for a guide, and `"No special requests"`.
They remain missing until the traveller confirms the refusal, at which point
the service asks one field-specific confirmation question without losing the
original stated preference.

An explicitly named itinerary city is enough evidence for a generic activity
such as `"Sightseeing in Jaipur"`; the service does not infer individual
attractions or experiences.

Traveller composition is reconciled only from explicitly stated adult, child,
and party counts. A total party size never implies that all travellers are
adults, and a stated child without an age is retained with `age: null`. When
explicit counts disagree, the stated composition is preserved and
`traveller_count_review` is added instead of silently changing the request.

Guide language or type is also evidence-bound: a generic guide request becomes
`"Yes"`; a language/type is emitted only when the traveller states it. Clear
activity/destination mismatches retain the requested activity and add
`activity_destination_review`, allowing downstream review without discarding a
user preference.

### Enquiry follow-ups

The request may include an optional `previous_extraction` containing the
`extraction` object returned for the preceding turn. The service extracts the
new message, overlays supplied scalar answers, and merges list answers without
duplicates. This retains previously captured trip details while a traveller
answers a clarification such as "no meals". The client owns this context; the
enquiry endpoint does not maintain a server-side conversation store.

### Redis response cache

`POST /api/v1/enquiry/understand` can reuse a successful, validated response
from Redis when `ENABLE_ENQUIRY_RESPONSE_CACHE=true`. Support also has a
separate, opt-in cache for strictly static RAG answers; live or mixed-source
support responses are never cached.

The cache key is a SHA-256 digest of a canonical, versioned payload containing
the whitespace-normalized message, normalized currency hint,
`previous_extraction`, configured model, `ENQUIRY_PROMPT_VERSION`, and
response-schema version. Message case and punctuation are preserved. The resulting key uses the
`swabi:cache:enquiry-understand:v1:` prefix, so raw enquiry text is not exposed
in Redis keys and prompt/schema changes naturally select a new entry.

Only successful responses that pass normal response validation and
post-processing are cached. Entries expire after
`ENQUIRY_RESPONSE_CACHE_TTL_SECONDS` (default 900 seconds). The request ID is
removed from the stored payload and replaced with the current request ID on
every cache hit.

On a miss, a short-lived Redis lock coordinates duplicate requests. A lock
owner rechecks the cache before calling the synchronous enquiry service;
waiters poll for the completed result and can proceed independently if the
owner does not finish within the configured wait. The defaults for both lock
TTL and wait are 210 seconds, chosen to cover the configured LLM timeout and
retry budget. Lock release uses a per-owner token check, preventing cleanup
from deleting a lock acquired by another request.

Redis read, write, and lock failures are fail-open: the request continues
through normal enquiry processing without relying on cache availability.
Readiness can report Redis degradation independently, and cache use resumes
when Redis recovers. Structured logs distinguish cache hits and misses and
record cache-operation failures; cache-specific metrics are not currently
exposed.

Verified checks for the enquiry cache cover Redis caching, request-ID
freshness, TTL expiration, single-flight behavior, safe lock cleanup, fail-open
operation during a Redis outage, and recovery after Redis restart.

### Date resolution

The prompt resolves:

- **Explicit dates** (`"15/12/26"`) → `dd/mm/yy`
- **Ordinal weekdays** (`"first Saturday of December"`, `"next Monday"`)
  → resolved using the current date
- **Ranges** → `travel_date` + `travel_date_end`
- **Duration** → preserved independently; it does not create an end date.

**Month-only expressions** (`"in December"`) leave both exact date fields
null. The date picker on the frontend collects the actual dates. The prompt
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

### AI Support

| Failure | Behavior |
|---|---|
| Missing or empty bearer token | 401 before support processing. |
| Private live API returns an error, malformed payload, or decryption error | Safe `needs_live_data` response; do not fabricate current facts. |
| Public catalogue API fails | Safe `needs_live_data` response; do not substitute stale catalogue data. |
| `CURRENT` missing, Qdrant unavailable, or alias mismatch on the RAG path | Fail closed; do not use an unverified knowledge version. |
| Retrieval fails or returns no evidence | `insufficient_context` with audience-specific app guidance. |
| Structured answer is truncated | Gateway retries once with an increased completion budget, capped at 4,096 tokens. |
| LLM answer cites no retrieved chunks or only unknown IDs | Fail closed to `insufficient_context`; mixed valid/invalid citations retain only valid IDs. |

Successful live API responses are passed through allowlisted field formatters
to the answer-generation LLM, which produces the user-facing response. List
limits are applied before generation and validated against the generated list;
if the model ignores the requested count, the service returns the trimmed
allowlisted facts. Live-only requests do not initialize RAG or perform a
knowledge-version check. Authentication and audience-scope checks remain
deterministic and run before any private API fetch.
The structured source planner runs for every support question and identifies
the supported live-data intent independently from any knowledge-base need.
Deterministic audience-scope checks remain in force.
Live API requests use an 8-second per-attempt timeout and allow up to
3 attempts. Retryable HTTP responses and transient transport failures
use bounded exponential retry delays, starting at 1 second.


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

---

## 15. AI Support and RAG (Phase 2)

### Endpoint and request path

`POST /api/v1/support/ask` accepts:

```json
{
  "question": "How can I book a package through an enquiry?",
  "audience": "customer"
}
```

In gateway mode, the route requires a verified NestJS principal assertion and
the `ai:support:ask` capability before any intent resolution, retrieval, or
answer generation. The assertion provides the trusted customer/vendor
audience; the request-body `audience` must match it. The original user
`Authorization: Bearer ...` token remains opaque and is passed explicitly to
the support service only for delegated live API calls. NestJS remains
responsible for resource authorization and business state.

### Server-managed conversation lifecycle

Multi-turn Support is available only when gateway authentication supplies a
verified principal subject. NestJS validates the user's JWT and constructs that
principal; FastAPI never trusts `user_id` or history provided by the request
body for conversation ownership.

```text
Browser -- JWT --> NestJS gateway -- signed principal + JWT --> AI Support
                                                                 |
                                 principal.sub + audience + conversation_id
                                                                 |
                                                                 v
                                                       Redis conversation list
                                                                 |
                                  bounded history ---------------+
                                                                 |
                                                    plan / live read / answer
                                                                 |
                                 atomically append user + assistant turns
```

1. A new request omits `conversation_id`; Support creates a UUID and starts
   with no history.
2. A follow-up provides the returned ID. Support derives a Redis key from the
   SHA-256 hash of the verified subject, verified audience, and ID, then loads
   the most recent configured exchanges.
3. After an answer is produced, one Redis transaction appends the user and
   assistant turns, trims the list to `SUPPORT_CONVERSATION_MAX_TURNS * 2`
   messages, and resets the inactivity expiry.
4. The response always returns the conversation ID for the next request.

The key format is
`swabi:support:conversation:v1:<user-sha256>:<audience>:<conversation-id>`.
It contains serialized `{role, content}` turns only: no raw user ID, JWT, or
gateway secret. Redis unavailability fails open: Support can still answer, but
the request has no persisted conversation context. A persistence failure is
logged for operations to investigate.

`AI_AUTH_MODE=legacy_service_token` deliberately yields no trusted principal
subject. It must therefore not be used to validate server-managed multi-turn
behavior; in that mode Support neither loads nor appends Redis conversation
history. Its request `conversation_history` field is transitional
compatibility only.

#### NestJS implementation responsibilities

For each Support request, NestJS must:

1. Validate the browser JWT and derive the user ID, audience, and permitted
   `ai:support:ask` scope.
2. Forward the original JWT in `Authorization` and attach the signed,
   short-lived principal headers described in
   [`auth_gateway_contract.md`](auth_gateway_contract.md).
3. Forward the request body's `conversation_id` unchanged when it is a
   follow-up; never accept a browser-supplied principal header.
4. Return the AI response's `conversation_id` to the client and reuse it only
   for the same signed-in user and audience.

The AI service enforces ownership by the scoped Redis key. A copied ID used by
another user or audience resolves to a different key and cannot reveal the
original conversation.

#### Operational acceptance check

After a gateway-mode first turn and one follow-up using the same returned
`conversation_id`, Redis should contain one conversation list with four
messages. Operators can inspect it without needing the raw user ID:

```redis
SCAN 0 MATCH swabi:support:conversation:v1:* COUNT 100
TYPE <key>
LLEN <key>
LRANGE <key> 0 -1
TTL <key>
```

Expected results are `list`, `4`, alternating serialized `user` and
`assistant` turns, and a positive TTL close to
`SUPPORT_CONVERSATION_TTL_SECONDS`. A later exchange increments the list by
two; after the configured limit is reached, the oldest exchange is trimmed.
The application startup log must also show that Redis connected. Do not use
`KEYS *` in production; `SCAN` is incremental and safe for this diagnostic.

### Routing and answer source

The service creates a structured source plan before fetching live data or
retrieving knowledge:

1. A structured LLM planner runs for every question and returns a supported
   live-data intent, its entities, and whether a separate knowledge query is
   needed. Mixed questions are not gated by keyword or phrase heuristics.
2. Deterministic checks enforce audience scope and validation. They do not
   authorize a request based on an LLM classification.
3. Audience checks prevent cross-role access; private intents require a JWT.
   NestJS remains authoritative for user authorization and business data.
4. Live responses are formatted to allowlisted fields and passed with the
   user's request and relevant conversation context to answer generation.
   The answer model is instructed to use the supplied live facts, not to
   treat itself as a source of current account data. Explicit list limits are
   enforced before generation and checked afterward; if generation violates
   a requested count, the service returns the trimmed verified facts.
5. When the planner identifies a knowledge need, the service retrieves
   approved RAG evidence using the focused knowledge query and includes it in
   grounded answer generation with citations. Unsupported
   questions fail safely with audience-specific guidance rather than
   invented facts.

Gateway-mode follow-ups load up to 10 recent user/assistant turns from the
server-managed Redis conversation list. Context helps interpret references and
constraints but cannot change the authenticated identity, audience, or access
scope. Broad payment-history requests may summarize up to 20 available
records when no count is specified; other unspecified lists default to five.
Live-only requests avoid RAG/Qdrant checks when the source plan identifies no
knowledge need. The source planner is invoked for both simple and compound
requests, including unfiltered catalogue requests.

Live transport currently resides in
`app/capabilities/support/live_data.py`. It calls NestJS, forwards the
original user JWT for a user-specific request, unwraps response envelopes,
and decrypts encrypted payloads when configured. It does not access
PostgreSQL. Public intents include package search/detail, public activity
search, and popular destinations; the outer Support request remains
gateway-authenticated even when the downstream catalogue API is public.
Customer/vendor private intents use their corresponding NestJS APIs.

The package list API currently supports `page`, `limit`, and `destination`.
It does not expose season, travel-date availability, budget, or group-size
filters. Support responses must not claim those constraints have been
applied or that matching availability is confirmed.

### Static RAG response cache

When `ENABLE_SUPPORT_RAG_RESPONSE_CACHE=true`, the support route may reuse
successful knowledge-base-only answers. The flag defaults to false. The route
still runs the source planner and verifies the current knowledge-index
authority before checking Redis; a cache hit skips retrieval and answer
generation, not the live-data decision.

A request is eligible only when the planner confidently selects no live-data
intent and requires RAG, deterministic checks find no live-data request, and
the request has no conversation history, conversation ID, or user ID. The
current knowledge version must also validate. Public catalogue requests,
private live-data requests, mixed-source answers, uncertain planner results,
fallbacks, and unsupported answers bypass caching.

The versioned SHA-256 key includes the normalized question, audience,
knowledge area and version, intent and answer prompt versions, configured
provider/model, response-schema version, and retrieval settings. It does not
include planner-generated query wording, so equivalent requests do not miss
because of harmless planner variation. Only an `answered` response with
valid knowledge citations, the verified version, and no live-data markers or
warnings can be stored. Request IDs are cleared before storage and set to the
current request on a hit.

Entries expire after `SUPPORT_RAG_RESPONSE_CACHE_TTL_SECONDS` (default 900
seconds). Redis single-flight locks use owner tokens and configurable TTL and
wait limits. Redis failures fail open to the normal support flow. Live-data
answers remain uncached regardless of this feature flag.

Targeted support-cache, support-route, and service tests passed on 19 Nov 2026
(33 passed).

### Multi-source execution flow

Support requests can use knowledge retrieval, live business data, or both.
The structured source plan independently identifies a live-data intent and
any knowledge-base query; phrase heuristics do not decide which source runs.

```text
                  Support Question
                         |
                         v
                Authentication and
                  Audience Checks
                         |
                         v
                   Intent Resolver
                         |
              +----------+----------+
              |                     |
              v                     v
        Live Intent?           Knowledge Required?
              |                     |
         +----+----+           +----+----+
         |         |           |         |
        Yes        No         Yes        No
         |         |           |         |
         v         |           v         |
   Authorized      |      Hybrid RAG     |
   NestJS Fetch    |      Retrieval      |
         |         |           |         |
         +----+----+-----------+---------+
              |
              v
       Answer Generation
       (available live facts,
        retrieved knowledge,
        conversation context)
              |
              v
        Response Validation
              |
              v
        Support API Response
```

Execution modes:

| Mode | Live fetch | RAG retrieval | Result |
|---|---|---|---|
| Knowledge-only | No | Yes | Grounded knowledge answer with citations. |
| Live-only | Yes | No | Answer generated from allowlisted live facts. |
| Mixed-source | Yes | Yes | Combined answer with live-data metadata and knowledge citations. |

For mixed-source requests, live-data retrieval and RAG retrieval are
independent evidence sources. The answer-generation layer receives both
when available and must use each source only for the information it supports.

Example:

`Show my latest bookings and explain the cancellation policy.`

- Live intent: `MY_BOOKINGS`
- Live endpoint: `GET /api/bookings/my`
- Knowledge requirement: cancellation policy
- RAG retrieval: approved cancellation-policy chunks
- Retrieval mode: `hybrid_live`

The service must not replace a failed personal-data lookup with
knowledge-base content or invent missing booking information.
```

### Add response metadata documentation

<CodeBlock language="markdown" editable>
### Source and retrieval metadata

The Support response exposes source information so callers can distinguish
knowledge-backed answers from live-data answers.

- `retrieval_mode_used`: identifies the retrieval mode. `hybrid_live`
  indicates that live data and knowledge retrieval were both used.
- `live_data_used`: true when live data was successfully incorporated.
- `live_data_sources`: identifies the NestJS endpoint or endpoints used.
- `live_data_warning`: exposes a live-data retrieval warning when present.
- `citations`: references the retrieved knowledge chunks supporting the answer.
- `knowledge_version`: identifies the knowledge index version used.
- `answer_type`: identifies the response outcome.

For a mixed-source response, `live_data_used` should be true,
`live_data_sources` should contain the successfully used endpoint labels,
and `citations` should contain valid knowledge references.

For a live-only response, knowledge retrieval is skipped and citations
remain empty. For a knowledge-only response, live-data metadata remains
unused.
```

### Customer enquiry-bid classification and retrieval

`MY_ENQUIRY_BIDS` is a customer-only live intent. It is intentionally checked
before the general `MY_ENQUIRIES` list intent: a question about bids must not
silently become a request to list enquiries. The deterministic bid patterns
cover direct forms (for example, "bids on my enquiry"), named forms (for
example, "bids on my Goa enquiry"), and aggregate forms (for example, "are
there bids on any of my enquiries?").

The classifier may attach either an `enquiry_id` (a UUID) or an
`enquiry_query` (an explicitly named title/destination). Generic qualifiers
such as `any`, `latest`, and `all my` are not treated as a named enquiry.

```text
Customer bid question
        |
        v
MY_ENQUIRY_BIDS classifier (before MY_ENQUIRIES)
        |
        +-- UUID ----------> GET /api/bids/enquiry/{enquiryId}
        |
        +-- named enquiry -> GET /api/enquiries/my
        |                         |
        |                         +-- one normalised title/destination match
        |                         |       -> GET /api/bids/enquiry/{enquiryId}
        |                         +-- zero or multiple matches
        |                                 -> deterministic not-found/disambiguation text
        |
        +-- general enquiry -> GET /api/enquiries/my
                                  -> sum `bidCount` for non-deleted enquiries
                                  -> deterministic aggregate summary
```

The aggregate route intentionally does not call the detailed bid endpoint for
every enquiry. It avoids an N+1 request pattern and reports only the counts
already present in the authorised enquiry list. The named route makes the
second call only after resolving exactly one enquiry ID. Formatter output is
allowlisted: detailed results contain bid amount, currency, vendor company,
status, negotiation status, and creation date; aggregates contain counts and
enquiry titles/destinations only.

This is live account retrieval, not RAG retrieval. RAG remains responsible
for stable bidding process and policy questions. The implementation assumes
that `GET /api/enquiries/my` provides the complete authorised enquiry set and
accurate `bidCount` values; if NestJS paginates or changes that field, the
support summary must be updated with explicit pagination/contract handling
before it can claim to cover all enquiries.

### Retrieval, policy lifecycle, and answers

The knowledge path:

1. Reads `data/rag_index/CURRENT` and confirms the Qdrant alias points at the
   same version. Missing state, Qdrant failure, or mismatch fails closed.
2. Filters chunks by audience, area, and lifecycle. Policy chunks must be
   approved, effective on the current date, and not expired. Lifecycle
   status is metadata in the YAML knowledge configuration and Qdrant; Qdrant
   payload indexes cover the `audience` and `area` filter fields. There is no
   AI-PostgreSQL policy registry.
3. Runs dense vector and BM25 keyword search over that eligible version and
   combines rankings with Reciprocal Rank Fusion (RRF).
4. Sends up to `SUPPORT_MAX_CONTEXT_CHUNKS` chunks (default 5), each limited
   by `SUPPORT_MAX_CHUNK_CHARS` (default 1,200), to structured answer
   generation.
5. Validates answer type and citations against retrieved chunk IDs. Invalid
   IDs are removed; an answer without any valid knowledge citation fails
   closed. Citation output is capped at three.

#### Qdrant Cloud diagnostics

Run `python scripts/diagnose_qdrant_latency.py` to inspect connectivity and
compare equivalent REST and gRPC queries against the configured Qdrant
endpoint. It resolves the diagnostic's physical collection from the configured
alias rather than relying on a version hardcoded in the script. This does not
replace the serving-time authority check: production retrieval still requires
`data/rag_index/CURRENT` and the Qdrant alias to identify the same published
version.

In the 2026-11-16 Qdrant Cloud run, the 404 was traced to an outdated
hardcoded collection version. REST and gRPC returned identical ordered chunk
IDs; REST had lower median and P95 latency, so the service continues to use
REST. These latency observations describe that run and endpoint, not a
transport guarantee. The diagnostic requires network access and valid
credentials for the configured Qdrant service, and generates a query embedding.

Structured answers use `SUPPORT_ANSWER_MAX_TOKENS` (default 1,536). The
Groq response `finish_reason` is retained; when it is `length` and the
structured output is invalid, the gateway retries with a larger completion
budget, capped at 4,096. Other invalid structured output receives the normal
single retry. This mitigates provider truncation but does not guarantee
provider availability or correctness.

### Reranking and current benchmark

An optional local cross-encoder can rerank the fused RRF candidates, but
serving defaults to `RAG_RERANKER_ENABLED=false`. The latest 15-case
comparison reported:

| Mode | Recall@5 | MRR@5 | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|
| Hybrid RRF | 0.973 | 0.950 | 0.937 | 2,135 ms |
| Hybrid + reranker | 0.940 | 0.922 | 0.911 | 3,636 ms |

Reranking regressed each reported retrieval metric and increased latency in
this run; do not enable it for production based on model availability alone.
See [`hybrid_retrieval.md`](hybrid_retrieval.md) and
`retrieval-reranked-phase2.json` for evaluation details.

The latest `grounding-report-phase2.json` predates the structured-output and
citation changes. It reported retrieval evidence recall 0.973, citation
precision 0.700, citation recall 0.716, and semantic judge coverage 0.467
(8 judge errors). Its answer-support rate of 1.0 applies only to the seven
judged cases. The grounding CLI records judge errors per case and continues
with later cases.

### Mixed-source regression tests

| Test question | Expected routing |
|---|---|
| Show my latest bookings | Live API only |
| Explain the cancellation policy | RAG only |
| Show my latest bookings and explain the cancellation policy | Live API + RAG |
| How can I cancel my booking? | RAG only |
| Show my booking status and explain refund rules | Live API + RAG |

Validation requirements:

- Verify the correct live-data intent is resolved.
- Verify `use_rag` is enabled for mixed-source requests.
- Verify the correct authenticated customer's data is returned.
- Verify the answer uses the retrieved policy rather than invented rules.
- Verify citations and live-data source metadata.
- Verify failed live-data requests do not produce fabricated personal data.
- Verify existing knowledge-only and live-only behavior remains unchanged.

### Phase 2 status

The RAG customer-support feature scope is complete for the current phase. The
platform overview knowledge update was published as version
`48fa84f6d107`, indexed in Qdrant, and smoke-tested; `CURRENT` and the Qdrant
alias matched, and the overview question returned the new grounded content.
The latest recorded full test run passed 769 tests with 44 deselected.

This feature-phase completion is not a blanket production certification. The
latest grounding report predates later answer-generation and support changes:
it reported semantic judge coverage of 0.467 (8 judge errors), citation
precision of 0.700, and citation recall of 0.716. Rerun grounding against the
current code and review its per-case results before using those metrics as a
release gate. Also verify real NestJS endpoint shapes, JWT forwarding,
decryption, empty results, and error behavior in the target environment, and
run deployment-specific latency/configuration/logging checks. Unit tests and
the local Qdrant smoke test do not establish those external contracts.

To publish a knowledge update, run `python -m scripts.rag_ingest --release`,
then index the reported version with
`python -m scripts.rag_index --version <version>`. Confirm that
`data/rag_index/CURRENT` and the Qdrant alias resolve to the same version
before accepting RAG traffic; the service fails closed when they disagree.
Coordinate these steps with deployment because the two authorities are
updated separately. The active development index when this status was
recorded was `48fa84f6d107`.
