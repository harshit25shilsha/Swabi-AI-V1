# Hybrid Retrieval in the RAG Pipeline

This guide describes the hybrid retrieval implementation used by Swabi AI:
BM25 keyword search and dense vector search, combined with Reciprocal Rank
Fusion (RRF). It covers the path from an incoming support question to the
ranked chunks passed to answer generation, the design choices behind that
path, and how to evaluate and tune it.

For the API and RAG service overview, see the [README](../README.md). The
implementation is in [`app/rag/retriever.py`](../app/rag/retriever.py) and
[`app/rag/qdrant_store.py`](../app/rag/qdrant_store.py); the benchmark is
implemented in [`app/evaluation/retrieval.py`](../app/evaluation/retrieval.py)
and [`scripts/evaluate_retrieval.py`](../scripts/evaluate_retrieval.py).

## What hybrid retrieval means

Retrieval-Augmented Generation (RAG) finds trusted source passages and gives
them to a language model as context. Retrieval quality matters: the answer
model cannot cite or use a relevant passage that retrieval did not return.

Hybrid retrieval combines two search methods that find relevance in different
ways:

| Method | What it compares | Usually good at |
|---|---|---|
| Dense vector search | Semantic similarity between an embedded question and embedded chunks | Paraphrases and related concepts |
| BM25 keyword search | Query terms and their importance in chunk text | Exact words, names, product terms, and phrases |

Neither method is a universal replacement for the other. A user may ask
about “getting paid,” while the source says “vendor payout” or “withdrawal.”
Dense search may connect those meanings. Conversely, an exact term such as
`lockedBalance` or “Swabi Plus” may be easier for keyword matching. Hybrid
search seeks to benefit from both.

## Components and responsibilities

### 1. Published knowledge chunks

The ingestion pipeline turns the approved knowledge source into chunks. Each
chunk has text and metadata such as `chunk_id`, `section_id`, `area`,
`audience`, and policy lifecycle fields. The indexer embeds each chunk and
stores its vector and payload—including the original text—in Qdrant.

The Qdrant alias (for example, `swabi_chunks`) points at a versioned physical
collection. The local `data/rag_index/CURRENT` file identifies the version
the application expects to serve.

### 2. Query embeddings

[`EmbeddingModel`](../app/rag/embeddings.py) uses FastEmbed with the
configured model (`BAAI/bge-small-en-v1.5` by default, 384 dimensions).
It embeds a query and normalizes its vector. At ingestion, the same model
embeds passages. Qdrant's cosine similarity ranks chunks by semantic
similarity to the query vector.

### 3. Dense search

`QdrantStore.search` sends the query vector to Qdrant. Qdrant applies the
shared eligibility filters, compares vectors, and returns its top-ranked
chunks with similarity scores. Dense search uses `RAG_MIN_SCORE` (default
`0.45`) as a score threshold; results below the threshold are omitted.

### 4. BM25 keyword search

`QdrantStore.keyword_search` scrolls the eligible chunk payloads from the
same Qdrant collection and builds a `rank_bm25.BM25Okapi` index over their
text. The tokenizer case-folds text and extracts Unicode word sequences
using `\w+`. BM25 ranks chunks based on query-term occurrence, term rarity
across the eligible corpus, and document length.

Only documents containing at least one query token are returned. BM25 scores
are useful for ordering the keyword results, but they are not cosine scores
and are not compared numerically to dense scores. No BM25 score threshold is
currently applied.

The corpus and BM25 index are cached in-process. The bounded LRU cache holds
up to 32 filter scopes and keys each entry by Qdrant URL, base collection,
resolved physical collection/version, audience, area, and UTC date. The date
is part of the key because policy eligibility can change when a policy
becomes effective or expires. The first request for a scope loads the
eligible corpus; later requests in that process reuse it. A new process or
cache eviction requires another initial load.

### 5. Reciprocal Rank Fusion

BM25 and dense search produce scores on unrelated scales. RRF avoids adding
or comparing those raw scores. It combines only each result's position
(rank) in its own list.

For a chunk \(d\), with one-based rank \(r_i(d)\) in each result list \(i\):

```text
RRF(d) = sum(1 / (c + r_i(d)))
```

The implementation uses a default rank constant `c = 60`. A chunk present in
both result lists gets contributions from both; a chunk appearing in only one
list gets one contribution. Higher fused scores rank first. Duplicate chunk
IDs are merged, and the score on the returned `RetrievedChunk` is replaced
with its fused score.

Example: if a chunk ranks fourth in dense results and second in BM25:

```text
RRF = 1 / (60 + 4) + 1 / (60 + 2)
    = 1 / 64 + 1 / 62
```

A chunk that appears only at rank one in dense search receives `1 / 61`.
The shared chunk can therefore outrank it because both methods support it.
These are ranking signals, not probabilities or calibrated confidence.

## End-to-end request workflow

1. **Resolve what kind of question this is.** The support service routes live
   business questions to authorized Swabi APIs when appropriate. RAG is for
   stable knowledge, not current bookings, wallet balances, or live bids.
   Successful live API responses are rendered from allowlisted fields by
   deterministic formatters; they do not depend on an LLM call. This keeps
   current facts grounded in the API and avoids spending Groq tokens to
   rephrase data the service already knows how to present.
2. **Verify knowledge authority.** Before answering from RAG, the service
   checks that the Qdrant alias points at the version named by local
   `CURRENT`. If they disagree, the service fails closed rather than serve
   stale policy.
3. **Pin retrieval to that version.** The verified version is passed to the
   retriever. Both searches target the same versioned collection, preventing
   dense and BM25 from mixing different published indexes.
4. **Prepare and embed the retrieval query.** Known general process patterns
   can be expanded into canonical retrieval wording; package-booking questions
   currently map to the documented enquiry/bidding workflow. The original
   question is still sent to the answer model. FastEmbed embeds the prepared
   query; an empty query returns no results and embedding failures become
   retrieval errors.
5. **Apply the same eligibility rules to both methods.** Qdrant filters
   audience, area, and current policy lifecycle before ranking:
   - A customer search matches `customer` and `both`; it excludes
     vendor-only chunks.
   - A vendor search matches `vendor` and `both`.
   - An area such as `swabi` or `travel` also matches chunks marked `both`.
   - Non-policy chunks marked `not_applicable` are eligible. Policy chunks
     must be approved, effective today, and not expired.
6. **Retrieve candidates from each method.** With requested result count
   `k`, each search requests `k * candidate_multiplier` candidates. The
   production default multiplier is 4. Dense search also applies its
   configured minimum score; BM25 ranks its filtered corpus.
7. **Fuse the ranked lists.** RRF combines the two candidate lists using the
   configured rank constant (60 by default), deduplicates by chunk ID, and
   returns at most `k` chunks.
8. **Generate and validate the answer.** The support service supplies the
   retrieved text to the answer model. It validates citations against the
   retrieved IDs and returns a safe insufficient-context response if
   retrieval fails or yields no chunks.

When retrieval cannot support an answer, the fallback suggests relevant
customer or vendor app areas (for example, vendors can check New Enquiries,
bids, Wallet/Transactions, Withdrawals, or Subscription) and directs policy
or unresolved account questions to Swabi support. These suggestions do not
replace a grounded policy answer.

The public support response reports `retrieval_mode_used="hybrid"` when the
knowledge base was searched. For answers supplied only by a live API, it
reports `"not_applicable"`; `live_data_used` and `live_data_sources` identify
that path. RRF scores are internal ranking scores; the response confidence
is a separate answer-generation field.

## Why these design choices

- **Use both lexical and semantic matching.** The query and approved source
  may use different wording, but exact business terminology also matters.
- **Fuse ranks, not raw scores.** Cosine similarity and BM25 scores have
  different meanings and ranges. RRF combines them without score
  normalization or a learned weighting model.
- **Filter before ranking.** Audience, area, and policy validity are
  correctness requirements, not relevance hints. Ineligible chunks should
  never reach the answer model.
- **Keep the version explicit.** A verified version shared by both methods
  prevents mixed-version retrieval and helps the cache naturally expire when
  a new collection is published.
- **Cache the BM25 corpus.** Rebuilding the keyword index on every request
  caused repeated full collection scans. A bounded process-local LRU removes
  that repeated cost while separating audience, area, version, and date
  scopes.
- **Retrieve more candidates than the final `k`.** Fusion needs room for a
  chunk that appears in one method but not near the top of the other.
  Candidate depth is a recall/latency trade-off.
- **Keep BM25 independently runnable.** This makes it possible to measure
  what each method contributes and diagnose misses rather than treating
  hybrid output as a black box.

## Evaluation and metrics

The labeled cases live in
[`app/evaluation/datasets/retrieval_v1.json`](../app/evaluation/datasets/retrieval_v1.json).
Each case contains a query, one or more relevant chunk IDs, audience, and
area. Multiple labels are appropriate when an answer spans several source
passages. Labels should be reviewed against the current knowledge source and
index whenever chunks change. The evaluation runners use the same deterministic
query preparation as production, so process-query rewrites are included in
retrieval metrics.

The evaluation runner compares `dense`, `bm25`, and `hybrid` search and
reports per-case rankings, metrics, and mean latency. With
`--include-reranked`, it also runs `hybrid_reranked` and reports aggregate
quality/latency deltas against `hybrid`:

- **Recall@k** = relevant returned chunks in the first `k` / all labeled
  relevant chunks. This measures coverage.
- **Precision@k** = relevant returned chunks in the first `k` / `k`. This
  measures how much of the result list is relevant. The denominator remains
  `k` even when fewer than `k` chunks are returned.
- **MRR@k** = reciprocal rank of the first relevant chunk, or zero if none
  appears in the first `k`. This rewards finding one good result early.
- **nDCG@k** = discounted cumulative gain divided by ideal discounted gain.
  The current labels use binary relevance: a labeled chunk is relevant; an
  unlabeled chunk is not. Results near the top count more than results lower
  in the list.
- **Mean latency** = average time spent in the search function per case for
  that mode. It is an observed benchmark, not a production latency
  guarantee.

Metrics are averaged across cases. If a case has multiple relevant chunks,
Precision@k and Recall@k are calculated per case and then averaged;
Recall@k and nDCG@k account for multiple relevant chunks, while MRR@k uses
only the first relevant result. Precision depends on complete relevance
labels: an unlabelled-but-relevant result is counted as non-relevant.

### Run the comparison

Install the RAG requirements (including `rank-bm25`), make sure Qdrant is
available, publish/index the knowledge base, and ensure the alias agrees
with local `CURRENT`. From the repository root:

```powershell
python scripts/evaluate_retrieval.py --k 5
python scripts/evaluate_retrieval.py --k 5 --output retrieval-report.json
```

The CLI checks index authority and records the knowledge version, embedding
model, dense score threshold, and fusion settings in its JSON output. It does
not modify production settings.

For the local cross-encoder mode, install the RAG requirements
(`fastembed>=0.9.0`) and run:

```powershell
python scripts/evaluate_retrieval.py --k 5 --include-reranked
```

The default model is FastEmbed's ONNX-compatible
`Xenova/ms-marco-MiniLM-L-6-v2`; its weights are downloaded on first use.
Production serving keeps reranking disabled unless
`RAG_RERANKER_ENABLED=true`. `RagRetriever` accepts a `Reranker` implementation
so a future deployment-specific cloud model can be injected without coupling
retrieval to a vendor.

### Tune fusion for experiments

The evaluator exposes two experimental parameters:

```powershell
python scripts/evaluate_retrieval.py `
  --k 5 `
  --rrf-rank-constant 20 `
  --candidate-multiplier 2
```

- A **smaller RRF rank constant** gives more influence to the difference
  between high ranks; a larger constant makes rank contributions closer
  together.
- A **larger candidate multiplier** gives RRF more opportunities to find
  relevant chunks at the cost of more search work and a larger fusion set.

These are evaluator arguments, not changes to production defaults. Compare
all metrics and inspect per-case ranked IDs; do not select a configuration
based on a single aggregate score.

## Grounding and citation evaluation

Run `python scripts/evaluate_grounding.py` against the same authoritative
index to execute the support-answer pipeline and measure retrieved evidence
recall, citation integrity, and citations' precision/recall against the
labeled evidence IDs. Add `--judge` to use the configured LLM provider to
score semantic answer support and citation support, and identify unsupported
claims. The judge receives each question, answer, and retrieved chunk text;
only use it with approved evaluation data. It incurs provider latency/cost.

The deterministic citation metrics do not claim to measure semantic support:
the optional judge supplies that assessment. Review per-case results and
expand/review the labeled evidence IDs as the knowledge base evolves. Provider errors are recorded per case in judge coverage while later cases
continue independently; deterministic metrics are retained for every case.
The judge output is kept deliberately short to reduce truncation risk.

## Current measured baseline

Earlier tuning runs against the 14-query dataset at `k=5` compared
these hybrid settings:

| Hybrid settings | Recall@5 | MRR@5 | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|
| RRF rank constant 60, candidate multiplier 4 (defaults) | 0.971 | 0.946 | 0.932 | about 2,119 ms |
| RRF rank constant 20, candidate multiplier 2 | 0.957 | 0.946 | 0.922 | about 2,124 ms |

These values are a point-in-time baseline, not a guarantee. The dataset is
small and its relevance labels need domain review. In this run the hybrid
missed at least one labeled chunk for `vendor-payouts`, even though the
retrieved chunks were useful for the broader question. This is a reason to
inspect labels and ranked IDs, not to assume a single parameter change will
solve the issue.

After adding the package-booking case and shared query preparation, a fresh
15-case hybrid run at `k=5` measured precision 0.320, recall 0.973, MRR 0.950,
and nDCG 0.937. The new booking case returned both labeled workflow chunks in
the first two positions. These are retrieval metrics only; they do not measure
generated-answer or citation support.

The evaluator also reports dense-only and BM25-only results for each run.
Use fresh CLI output when comparing all three modes; these hybrid-only
tuning results should not be used to infer the standalone modes' scores.

Latency also has a cold/warm-cache caveat: the first BM25 query for a filter
scope loads its corpus, while later queries reuse the process-local index.
The evaluation runs modes in sequence with a shared store, so its BM25 and
hybrid latency numbers do not have identical cache-warming conditions.
Run repeated measurements or compare cold and warm runs separately before
making performance claims.

### Latest Phase 2 comparison

The 2026-10-30 `retrieval-reranked-phase2.json` run uses 15 cases at `k=5`.
Hybrid RRF measured recall 0.973, MRR 0.950, nDCG 0.937, and mean latency
2,135 ms. The reranked run measured recall 0.940, MRR 0.922, nDCG 0.911,
and 3,636 ms. In this newer comparison reranking regressed every reported
quality metric and increased latency, so keep `RAG_RERANKER_ENABLED=false`.
This supersedes the earlier 14-case reranker comparison below for the Phase 2
release decision.

The matching `grounding-report-phase2.json` was an incomplete judge run:
judge coverage was 0.467 after structured-output failures, citation precision
was 0.700, and citation recall was 0.716. Treat its 1.0 answer-support rate
as applying only to the seven judged cases, not as a full-dataset result.
Rerun grounding after the completion-budget and citation-context changes
before using it as a release-quality baseline.

### Local cross-encoder result

On authoritative index `a441a35b9643` at `k=5`, the local
`Xenova/ms-marco-MiniLM-L-6-v2` reranker was compared with hybrid RRF:

| Mode | Precision@5 | Recall@5 | MRR@5 | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|---:|
| Hybrid RRF | 0.314 | 0.971 | 0.946 | 0.932 | 2,109 ms |
| Hybrid RRF + cross-encoder | 0.314 | 0.971 | 0.964 | 0.954 | 4,358 ms |
| Delta | 0.000 | 0.000 | +0.018 | +0.022 | +2,250 ms |

Reranking improved MRR and nDCG slightly without changing precision or recall,
but more than doubled mean latency in this run. Keep serving reranking disabled
until the product accepts that trade-off; these results cover only 14 labeled
queries and one index version. The report's per-case rankings show both gains
and regressions: `vendor-withdrawal-steps` MRR improved from 0.25 to 1.0, while
`booking-dispute` dropped from 1.0 to 0.5. Review the per-case rankings before
making a release decision.

## Practical interpretation

Suppose a vendor asks, “How do I get my money after a completed trip?”

- **Dense search** can connect “get my money” to the escrow and payout
  concepts even if those words do not match exactly.
- **BM25** can find passages that literally mention `withdrawal`,
  `escrowBalance`, or “payout.”
- **RRF** promotes passages supported by both methods and can retain useful
  passages unique to either list.
- **Metadata filters** ensure the answer is based on vendor-visible,
  current knowledge for the requested area.
- **Citation validation** ensures the answer only cites chunks actually
  returned to the model.

Retrieval itself does not decide which facts are true or authorize a
transaction. The approved knowledge base is the source for stable policy;
live data and actions remain with the Swabi backend.

## Implementation map

| Concern | Location |
|---|---|
| Query embedding and dense/BM25 orchestration | [`app/rag/retriever.py`](../app/rag/retriever.py) |
| Qdrant filters, vector search, BM25 search, BM25 cache | [`app/rag/qdrant_store.py`](../app/rag/qdrant_store.py) |
| Embedding model wrapper | [`app/rag/embeddings.py`](../app/rag/embeddings.py) |
| Retrieval query preparation and support integration | [`app/capabilities/support/retrieval_query.py`](../app/capabilities/support/retrieval_query.py), [`app/capabilities/support/service.py`](../app/capabilities/support/service.py) |
| Ranking metrics and evaluation report | [`app/evaluation/retrieval.py`](../app/evaluation/retrieval.py) |
| Evaluation CLI and mode wiring | [`scripts/evaluate_retrieval.py`](../scripts/evaluate_retrieval.py) |
| Local reranker and injectable interface | [`app/rag/reranker.py`](../app/rag/reranker.py) |
| Grounding and citation metrics | [`app/evaluation/grounding.py`](../app/evaluation/grounding.py) |
| Grounding evaluation CLI | [`scripts/evaluate_grounding.py`](../scripts/evaluate_grounding.py) |
| Labeled evaluation queries | [`app/evaluation/datasets/retrieval_v1.json`](../app/evaluation/datasets/retrieval_v1.json) |
| BM25 dependency | [`requirements-rag.txt`](../requirements-rag.txt) |
