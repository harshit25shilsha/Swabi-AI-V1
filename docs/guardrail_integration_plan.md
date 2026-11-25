# Guardrail Integration Plan

**Project:** Swabi AI
**Scope:** Integrate the existing Guardrail Service (GDv1) into the Swabi AI
Foundation (Phase 0) as a dedicated Chat Safety capability.
**Status:** Audit complete — ready for implementation.
**Revision:** v2 (incorporates the design review feedback).
**Branch:** `feat/guardrail-integration`

---

## 0. Purpose

This document is the Phase 1 audit deliverable and the implementation
specification. It supersedes the first draft. Where a section is unchanged
from v1, it is referenced rather than restated. Sections 2, 5, 6, and 7 are
the authoritative versions.

Two milestones are deliberately separated:

1. **Integration completion** — the code is consolidated and tested.
2. **Production release** — the failure policy, auth boundary, and
   deployment assumptions have been reviewed and approved.

Integration can be marked complete while production gates remain open.
See §7 for the release gate list.

---

## 1. Scope and locked assumptions

### In scope

- Integrate GDv1 as `app/capabilities/chat_safety/` inside the Phase 0
  FastAPI application (same repo, single `uvicorn`, one config).
- Consolidate shared infrastructure: config, logging, exceptions,
  observability, LLM gateway.
- Preserve the existing public contract for
  `POST /api/v1/validate-message`.
- Introduce a small, reusable result shape that future capabilities can
  adopt later — but do not introduce a framework now.
- Verify the auth boundary between the guardrail service token and the
  user-JWT context used for outbound Swabi calls.
- Preserve existing deterministic detection, prompt, and policy behavior.

### Locked assumptions

1. **Same repository, single FastAPI app.** Not a separate microservice.
2. **Public contract stays backward-compatible.** Same path, same request
   fields, same response fields, same `ALLOW`/`BLOCK` semantics, same
   category names. "Backward-compatible" means NestJS requires no code
   change; JSON whitespace and key ordering are not part of the contract.
3. **Inbound auth stays `Authorization: Bearer <GUARDRAIL_API_TOKEN>`.**
   Tightening the comparison to constant-time is in scope. Changing the
   mechanism is not.
4. **No user JWTs reach the guardrail.** User context
   (`sender_id`, `sender_role`, `conversation_id`) arrives as plain data
   from the already-authenticated NestJS caller. This is enforced, not
   merely documented. See §2.

---

## 2. Auth boundary (critical)

### 2.1 The problem

The Phase 0 `auth_context_middleware` currently captures **any**
`Authorization: Bearer …` header into `user_jwt_var`. That variable is
consumed by `SwabiClient._headers()` when the AI service calls NestJS on
behalf of a user.

The guardrail receives a **service token** in the same `Authorization`
header. If a future change inside the guardrail handler were to call
`SwabiClient`, the service token would be forwarded to NestJS as if it
were a user's JWT.

This is a latent leak today and becomes a real one the moment any
guardrail code path touches `SwabiClient`. It must be closed
structurally, not by convention.

### 2.2 Fix — path-based allow-list

The middleware must only capture the user JWT for routes that legitimately
act on behalf of a user. Service-to-service routes are excluded.

```python
# app/main.py

# Routes that legitimately carry a user JWT intended for outbound Swabi calls.
# Service-to-service routes are intentionally absent.
USER_JWT_ROUTE_PREFIXES = (
    "/enquiry",
    "/assistant",
    # future user-facing routes added here as they land
)


def _extract_bearer_token(auth_header: str) -> str | None:
    if not auth_header:
        return None
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = value.strip()
    return token or None


@app.middleware("http")
async def auth_context_middleware(request: Request, call_next):
    path = request.url.path

    if any(path.startswith(p) for p in USER_JWT_ROUTE_PREFIXES):
        auth_header = request.headers.get("Authorization", "")
        token = _extract_bearer_token(auth_header)
        set_user_jwt(token)
        set_user_id(request.headers.get("X-User-Id"))
    else:
        # Explicitly clear on service-to-service paths so a mixed-route
        # process cannot leak context across requests.
        set_user_jwt(None)
        set_user_id(None)

    return await call_next(request)

Two properties that must hold:

Allow-list, not deny-list. New routes default to no capture. Opt in
explicitly when a route is genuinely user-facing.

Explicit clear in the else branch. Do not rely on contextvar default
state; overwrite it every request.

2.3 Regression tests (required)
Add tests/integration/test_auth_boundary.py:

python
def test_guardrail_route_does_not_capture_user_jwt(monkeypatch):
    captured = {}

    async def fake_get_customer_profile(self, user_id):
        from app.core.auth_context import get_user_jwt
        captured["jwt"] = get_user_jwt()
        return {}

    monkeypatch.setattr(
        SwabiClient, "get_customer_profile", fake_get_customer_profile
    )

    client.post(
        "/api/v1/validate-message",
        headers={"Authorization": "Bearer guardrail-token-value"},
        json={"message": "hello"},
    )

    assert captured.get("jwt") is None
Plus a direct unit test asserting get_user_jwt() is None inside a
mocked guardrail handler that reads it. This second test protects against
the middleware being bypassed or reordered in the future.

2.4 What is explicitly not the fix
Adding a comment that says "the guardrail does not use user JWTs."

Relying on the guardrail handler to call set_user_jwt(None) itself.

Introducing a new header or scheme.

3. Existing architecture (summary)
Phase 0 — AI Foundation
A production-ready MVP FastAPI service with:

app/main.py — app, middleware, exception handlers, Redis lifespan.

app/core/ — config.py, logging.py, exceptions.py, redis.py,
auth_context.py.

app/llm/ — gateway.py, groq.py, schemas.py.

app/graph/ — minimal LangGraph foundation.

app/rag/ — Retriever protocol + MockRetriever.

app/swabi/ — SwabiClient with user-JWT forwarding.

app/observability/ — logging.py helpers.

app/evaluation/ — dataset loader and runner.

app/api/routes/health.py — /health, /ready.

23 passing tests under tests/unit/ and tests/integration/.

GDv1 — Guardrail Service
A standalone synchronous moderation service. Two-layer detection:
deterministic rules first, LLM classifier second, policy layer applies the
final decision.

app/api/moderation.py — POST /api/v1/validate-message.

app/api/metrics.py — /health, /metrics (Prometheus text).

app/detectors/ — 10 pure functions.

app/llm/providers.py — LLMProvider ABC, Groq and Gemini providers.

app/llm/moderator.py — SYSTEM_PROMPT, PROMPT_VERSION, provider
circuits, JSON parsing, consensus loop.

app/models/moderation.py — request/response/LLM schemas.

app/policies/policy.py — BLOCKED_CATEGORIES, apply_policy.

app/services/moderation.py — orchestrator.

app/observability/ — counters, circuit breaker, middleware, JSON
logging.

17 test files. Some naming/duplication issues to be verified before
fixing (see §5.7).

The asymmetry that matters
Concern	Phase 0 Foundation	GDv1 Guardrail
Caller	Frontend (via NestJS)	NestJS only
Auth direction	Inbound user JWT, outbound JWT forward	Inbound service token
Acts as user?	Yes	No
Talks to Swabi APIs?	Yes	No
This asymmetry is preserved by §2.

4. Components reused unchanged
From GDv1 — moved verbatim
All 10 detectors in app/detectors/. Detection logic unchanged.

app/policies/policy.py — BLOCKED_CATEGORIES and apply_policy.
The separation between detection, classification, and policy is
preserved exactly.

SYSTEM_PROMPT and PROMPT_VERSION.

app/models/moderation.py schemas, including all response fields.

app/observability/counters.py and circuit_breaker.py, moved to a
shared location.

app/observability/middleware.py event shape; the moderation log line
keeps its fields.

The GDv1 test suite, subject to the verification step in §5.7.

From Phase 0 — unchanged
Middleware ordering rules.

config.py structure (extended, not replaced).

exceptions.py hierarchy (extended).

auth_context.py (behavior changed for the guardrail path only, per §2).

llm/gateway.py (used by the guardrail via a small adapter).

swabi/client.py, rag/, graph/, evaluation/ — untouched.

5. Adaptations required
Each is small and localized.

5.1 Configuration — one Settings
Extend app/core/config.py with the guardrail's additional fields:

GUARDRAIL_API_TOKEN

GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT_SECONDS

LLM_FALLBACK_ENABLED

LLM_CONSENSUS_ATTEMPTS — default 1 (see §5.2)

LLM_CIRCUIT_FAILURE_THRESHOLD, LLM_CIRCUIT_OPEN_SECONDS

GROQ_MODEL, GROQ_TIMEOUT_SECONDS

LLM_TEMPERATURE, LLM_MAX_TOKENS

SERVICE_VERSION, LOG_LEVEL, LOG_FORMAT

Conventions:

Prefer lower-case Pydantic field names for new fields, matching the
Phase 0 convention. Add @property aliases where existing guardrail
code reads upper-case names, so no rename sweep is required.

Keep required-with-no-default for genuinely required secrets
(GROQ_API_KEY, GUARDRAIL_API_TOKEN). Provide development-safe
defaults for everything else so pytest and uvicorn start cleanly.

Fix the two invalid Python-syntax lines in .env.example
(LLM_CIRCUIT_FAILURE_THRESHOLD: int = 5,
LLM_CIRCUIT_OPEN_SECONDS: int = 60) — these break dotenv parsing.

5.2 LLM consensus — default to one attempt
Consensus makes multiple classification calls per message, increasing
latency, token cost, and failure surface. Preserve the code path but
change the default:

LLM_CONSENSUS_ATTEMPTS defaults to 1.

The multi-attempt loop remains in the code for future use.

Raising the default above 1 requires evaluation evidence that it
materially improves classification accuracy.

The existing BLOCK short-circuit and highest-confidence-ALLOW behavior
stay in place when the value is greater than 1.

5.3 LLM access — via the shared gateway
The guardrail's GroqProvider / GeminiProvider become thin adapters
over LLMGateway. Two shapes were considered:

(a) Adapter inside chat_safety/llm_adapter.py that calls
LLMGateway.generate(..., json_mode=True).

(b) Extend LLMGateway with a classify(system, user) convenience
method.

Chosen: (a) for this phase. Smaller surface, easier to revert.

Provider fallback (Groq → Gemini) is preserved inside the adapter.
Circuit breakers stay per-provider, using the shared CircuitBreaker
class. The prompt and consensus loop stay in chat_safety/; they do not
move into the gateway.

Note on the Groq client. Phase 0 uses raw httpx against the Groq
REST API. GDv1 uses the official groq Python SDK. Keep the httpx path
for consistency with the shared gateway. Drop groq==0.11.0 from
requirements.txt after the adapter is verified. google-genai stays
for the Gemini fallback adapter.

5.4 Logging — merge required functionality only
The GDv1 JsonFormatter is more capable than Phase 0's current setup, but
"more capable" is not sufficient reason to switch. Before adopting:

Compare both formatters.

Preserve every field currently emitted by Phase 0 logs.

Preserve the guardrail moderation event field set.

Verify request-id correlation still works in both directions.

Adopt only the functionality actually required for both paths. If a
merge produces a smaller formatter than GDv1's, prefer the smaller one.

5.5 Middleware — order verified by tests, not by reading
FastAPI/Starlette middleware wrapping is LIFO on request and FIFO on
response. Registration order and observed order differ. Do not assume.

Target behavior:

On request: request-id → auth-context → observability → route.

On response: route → observability → auth-context → request-id.

Add tests/integration/test_middleware_order.py that:

Asserts request_id_var is populated before the observability
middleware logs.

Asserts X-Request-ID is present on the final response.

Asserts auth_context_middleware runs on the guardrail route and
clears the JWT context.

If the observed order differs from the target, adjust the registration
until the tests pass. Do not change the tests to match the code.

5.6 Exceptions
Extend app/core/exceptions.py:

GuardrailError(SwabiAIError) — base for guardrail issues.

GuardrailUnavailableError(GuardrailError) — provider ladder
exhausted or all circuits open.

GuardrailValidationError(GuardrailError) — invalid request payload.

CircuitOpenError — subclass of GuardrailUnavailableError, kept so
existing except clauses continue to work.

5.7 The all circuits open bug
app/llm/moderator.py::_classify_with_fallback currently ends with
assert last_exc is not None. When every provider circuit is open and no
provider call is attempted, this raises AssertionError, and the
intended except CircuitOpenError branch in service.py is dead code.

Fix: replace the assertion with

python
raise CircuitOpenError("all provider circuits are open")
Caller-visible behavior is unchanged (both paths fail open), but the
intended path is now reachable and the signal is correct.

5.8 No-database import guard — scoped
An import-guard test that fails if psycopg, asyncpg, sqlalchemy, or
typeorm appear anywhere in the repo is too broad. The AI Foundation may
legitimately need a database client for a future capability.

Scope the guard to app/capabilities/chat_safety/ and its dependency
graph:

python
# tests/unit/chat_safety/test_no_db_imports.py
FORBIDDEN = {"psycopg", "psycopg2", "asyncpg", "sqlalchemy", "typeorm"}

def test_chat_safety_does_not_import_db_libraries():
    import ast
    from pathlib import Path
    root = Path("app/capabilities/chat_safety")
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert n.name.split(".")[0] not in FORBIDDEN, f"{py}: {n.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN, f"{py}: {node.module}"
5.9 Guardrail interface — deferred
Do not introduce a Guardrail ABC in this phase. Chat Safety is the only
consumer.

Introduce a small, plain Pydantic result shape that future capabilities
can reuse when they arrive:

python
# app/capabilities/chat_safety/schemas.py
class GuardrailDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"           # internal only
    UNAVAILABLE = "UNAVAILABLE" # internal only

class GuardrailResult(BaseModel):
    decision: GuardrailDecision
    reason_codes: list[str] = []
    confidence: float | None = None
    source: str
    message: str | None = None
The public endpoint continues to serialize ModerationResponse
(ALLOW/BLOCK only). REVIEW and UNAVAILABLE are internal and are
mapped before serialization.

When a second capability needs a shared contract, promote this shape to
app/guardrails/contracts.py and add an interface at that time. Not
before.

5.10 Test-suite verification before fixes
Several issues were reported in the earlier review of the GDv1 test
suite. Verify each is real before changing anything.

Reported items to verify:

tests/moderation.py missing test_ prefix.

Duplicate function names in tests/test_number_words.py.

assert caught_by_rules or True in tests/test_disguised_numbers.py.

Duplicate list entries in tests/test_integration_llm.py.

For each: confirm the issue exists, confirm the fix, apply the fix only
if real. Do not make speculative edits.

6. Target architecture
6.1 Directory layout
text
swabi-ai/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   ├── redis.py
│   │   └── auth_context.py
│   ├── llm/
│   │   ├── gateway.py
│   │   ├── groq.py
│   │   └── schemas.py
│   ├── capabilities/
│   │   └── chat_safety/
│   │       ├── __init__.py
│   │       ├── detectors/
│   │       ├── policies/policy.py
│   │       ├── prompts.py
│   │       ├── schemas.py
│   │       ├── llm_adapter.py
│   │       ├── service.py
│   │       └── dependencies.py
│   ├── observability/
│   │   ├── counters.py
│   │   ├── circuit_breaker.py
│   │   ├── middleware.py
│   │   └── logging.py
│   ├── swabi/
│   ├── rag/
│   ├── graph/
│   ├── evaluation/
│   └── api/routes/
│       ├── health.py
│       ├── metrics.py
│       └── chat_safety.py
├── tests/
│   ├── unit/
│   │   └── chat_safety/
│   ├── integration/
│   │   ├── test_health.py
│   │   ├── test_chat_safety_api.py
│   │   ├── test_auth_boundary.py
│   │   └── test_middleware_order.py
│   └── contract/
│       └── test_chat_safety_contract.py
├── docs/
│   ├── architecture.md
│   └── guardrail_integration_plan.md
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
app/guardrails/ is not introduced in this phase. See §5.9.

6.2 Request flow
text
NestJS
  │ POST /api/v1/validate-message
  │ Authorization: Bearer <GUARDRAIL_API_TOKEN>
  │ {"message": "...", "sender_id": "...", "conversation_id": "..."}
  ▼
FastAPI app
  ├─ request_id_middleware           (Phase 0)
  ├─ auth_context_middleware         (Phase 0 — clears JWT for this path, per §2)
  ├─ observability_middleware        (GDv1 — log event + counters)
  ▼
chat_safety route
  ├─ verify_token (secrets.compare_digest)
  ├─ ChatSafetyService.moderate()
  │    ├─ normalize
  │    ├─ deterministic detectors
  │    ├─ if no match: ChatSafetyLLMAdapter.classify()
  │    │      └─ LLMGateway.generate(json_mode=True) with provider ladder
  │    ├─ apply_policy
  │    └─ return ChatSafetyResult
  ├─ map to ModerationResponse (public contract)
  └─ attach request_id, prompt_version, provider
  ▼
observability_middleware emits moderation log line + counters
  ▼
NestJS receives the existing JSON shape
7. Risks, mitigations, and production gates
7.1 Risks and mitigations
Risk	Mitigation
Guardrail token captured as user JWT	Path-based allow-list in auth_context_middleware (§2) + test_auth_boundary.py
Fail-open policy shipped without sign-off	Integration preserves behavior; production release gate (§7.2) requires written approval
Consensus latency/cost	Default LLM_CONSENSUS_ATTEMPTS=1; raise only with evaluation evidence
No-DB test too broad	Scoped to app/capabilities/chat_safety/ (§5.8)
Premature guardrail abstraction	ABC deferred (§5.9)
Middleware order assumed	test_middleware_order.py asserts actual order (§5.5)
Speculative test fixes	Verification step required before editing (§5.10)
Token rotation undefined	README section on secret storage and rotation procedure
Fallback untested in CI	Mocked provider-fallback tests required; live tests optional
Per-process metrics / circuit state	Documented in README; worker count is a deployment decision
Sensitive content in logs	Truncated SHA-256 hash preserved; test asserts raw message never appears in captured logs
Public contract drift	tests/contract/test_chat_safety_contract.py asserts exact endpoint path, request fields, response fields, and ALLOW/BLOCK semantics
7.2 Production release gates
Integration completion and production release are separate milestones.
Production release requires all of the following.

□ Fail-open policy approved in writing by the Swabi
product/backend owner.
□ Auth boundary test (test_auth_boundary.py) passing in CI.
□ Contract test (test_chat_safety_contract.py) passing.
□ Token rotation procedure documented and rehearsed once on staging.
□ Deployment worker count decided and documented.
□ Gemini fallback verified by a mocked test; optional live
verification on staging.
□ Failure modes tested end-to-end: all providers down, invalid JSON,
timeout, circuit open.
□ Sensitive data verified absent from logs during the integration
test run.
□ Rate-limit expectation confirmed with the NestJS team.
□ Manager sign-off on the failure policy and the auth boundary.
8. Implementation sequence
Each step ends with a test run and a commit. Order is deliberate.

Branch. feat/guardrail-integration. Add this document. Commit.

Config merge. Extend app/core/config.py. Update .env.example.
Fix invalid lines. Add config unit tests. LLM_CONSENSUS_ATTEMPTS
defaults to 1. Commit.

Logging merge. Compare formatters; merge only required
functionality. Preserve all existing log fields. Commit.

Exceptions. Add GuardrailError, GuardrailUnavailableError,
GuardrailValidationError, CircuitOpenError. Commit.

Move shared observability. Move counters.py and
circuit_breaker.py into app/observability/. Run tests. Commit.

Verify test-suite issues. Enumerate the reported duplicates and
ineffective assertions. Confirm each is real. Fix only the confirmed
ones. Commit.

Move chat safety. Move detectors, policy, prompt, schemas,
service, dependencies into app/capabilities/chat_safety/. Move
tests into tests/unit/chat_safety/ and
tests/integration/test_chat_safety_api.py. Commit.

Add GuardrailResult. Introduce the internal result shape in
chat_safety/schemas.py. Do not add an ABC. Commit.

LLM adapter. Add chat_safety/llm_adapter.py that calls
LLMGateway. Update service.py to use it. Fix the
_classify_with_fallback assertion. Delete the old
providers.py after the adapter is verified. Commit.

Register routes and middleware. Add api/routes/metrics.py and
api/routes/chat_safety.py. Update app/main.py with the
USER_JWT_ROUTE_PREFIXES allow-list and middleware registration
order. Commit.

Add contract, auth-boundary, middleware-order, and no-DB tests.
Commit.

Mocked fallback test. Add
tests/unit/chat_safety/test_provider_fallback_mocked.py. Commit.

Docs. Add docs/architecture.md and the Chat Safety section in
README.md, including auth boundary, fail-open policy status
(pending approval), and token rotation. Commit.

Full suite. Run pytest, ruff check ., ruff format --check ..
Record actual results. Commit.

Final engineering report. Deliver per §10.

9. Definition of done (integration phase)
□ Single FastAPI app with /health, /ready, /metrics,
POST /api/v1/validate-message.
□ POST /api/v1/validate-message backward-compatible with today's
schema (verified by contract test).
□ Chat Safety under app/capabilities/chat_safety/.
□ auth_context_middleware does not capture the guardrail token
(verified by test).
□ GuardrailResult present as a small Pydantic model. No ABC.
□ LLM calls go through LLMGateway. No second Groq client.
□ CircuitOpenError raised when all circuits are open.
□ verify_token uses secrets.compare_digest.
□ LLM_CONSENSUS_ATTEMPTS defaults to 1; multi-attempt path preserved.
□ No-DB import test scoped to chat_safety/.
□ Middleware order verified by integration test.
□ Mocked provider-fallback test present.
□ pytest, ruff check ., ruff format --check . clean.
□ README.md and docs/architecture.md cover Chat Safety, auth
boundary, fail-open policy (pending approval), token rotation,
per-process metrics.
□ Final engineering report delivered.
10. Final engineering report — required contents
What was inspected.

What was reused (detectors, policies, prompts, schemas, gateway,
observability).

What changed (additions, modifications, deletions).

Architecture after integration: request flow and responsibilities.

API compatibility: actual endpoint and any contract changes.

Security and failure behavior: auth, timeouts, provider failures,
fallback policy.

Test results: commands, pass/fail counts, known failures.

Configuration changes: new or modified environment variables.

Remaining limitations and open decisions.

Next recommended phase.

Separate implemented functionality from recommendations and future work.
Do not mark the phase production-ready based on integration alone.

11. Explicitly out of scope
Full AI Enquiry Understanding.

RAG ingestion, embeddings, vector database.

Bid analysis, semantic matching.

Travel Assistant workflows, autonomous agents.

Direct database access from Python.

A separate microservice for every capability.

Rewriting the guardrail's prompt, detectors, or policy.

Changing the public moderation contract.

Changing the inbound auth mechanism.

A dynamic plugin registry or guardrail framework.