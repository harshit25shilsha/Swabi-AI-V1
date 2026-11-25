"""Offline ranking evaluation for RAG retrieval modes."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import log2
from pathlib import Path

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvalDataset
from app.rag.schemas import RetrievedChunk


class RetrievalEvalCase(BaseModel):
    id: str
    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    audience: str = "customer"
    area: str | None = None


class RetrievalEvalDataset(BaseModel):
    name: str
    cases: list[RetrievalEvalCase] = Field(min_length=1)


def load_retrieval_dataset(path: str | Path) -> RetrievalEvalDataset:
    """Load the project evaluation format and map it to retrieval labels."""
    dataset = EvalDataset.load(path)
    return RetrievalEvalDataset.model_validate(
        {
            "name": dataset.name,
            "cases": [
                {
                    "id": case.id,
                    "query": case.input,
                    **case.expected,
                    **case.metadata,
                }
                for case in dataset.cases
            ],
        }
    )


@dataclass
class RetrievalCaseResult:
    case_id: str
    query: str
    ranked_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    latency_ms: float


@dataclass
class RetrievalModeReport:
    mode: str
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    mean_latency_ms: float
    cases: list[RetrievalCaseResult]


@dataclass
class RetrievalEvalReport:
    dataset: str
    k: int
    modes: dict[str, RetrievalModeReport]

    def to_dict(self) -> dict:
        result = {
            "dataset": self.dataset,
            "k": self.k,
            "modes": {name: asdict(report) for name, report in self.modes.items()},
        }
        baseline = self.modes.get("hybrid")
        reranked = self.modes.get("hybrid_reranked")
        if baseline is not None and reranked is not None:
            result["comparisons"] = {
                "hybrid_reranked_vs_hybrid": {
                    metric: getattr(reranked, metric) - getattr(baseline, metric)
                    for metric in (
                        "precision_at_k",
                        "recall_at_k",
                        "mrr_at_k",
                        "ndcg_at_k",
                        "mean_latency_ms",
                    )
                }
            }
        return result


SearchFn = Callable[[RetrievalEvalCase, int], list[RetrievedChunk]]


def ranking_metrics(
    relevant_chunk_ids: list[str],
    ranked_chunk_ids: list[str],
    *,
    k: int,
) -> dict[str, float]:
    """Compute binary-relevance Precision@k, Recall@k, MRR@k, and nDCG@k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_chunk_ids)
    if not relevant:
        raise ValueError("at least one relevant chunk is required")

    unique_ranked: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranked_chunk_ids:
        if chunk_id not in seen:
            unique_ranked.append(chunk_id)
            seen.add(chunk_id)
    ranked = unique_ranked[:k]
    hits = [1 if chunk_id in relevant else 0 for chunk_id in ranked]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant)
    reciprocal_rank = next(
        (1 / rank for rank, hit in enumerate(hits, start=1) if hit),
        0.0,
    )
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate_retrieval(
    dataset: RetrievalEvalDataset,
    searchers: Mapping[str, SearchFn],
    *,
    k: int = 5,
) -> RetrievalEvalReport:
    """Run each dataset query through every search mode and aggregate metrics."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not searchers:
        raise ValueError("at least one retrieval mode is required")

    reports: dict[str, RetrievalModeReport] = {}
    for mode, search in searchers.items():
        results: list[RetrievalCaseResult] = []
        for case in dataset.cases:
            start = time.perf_counter()
            chunks = search(case, k)
            latency_ms = (time.perf_counter() - start) * 1000
            ranked_ids = [chunk.chunk_id for chunk in chunks]
            metrics = ranking_metrics(
                case.relevant_chunk_ids,
                ranked_ids,
                k=k,
            )
            results.append(
                RetrievalCaseResult(
                    case_id=case.id,
                    query=case.query,
                    ranked_chunk_ids=ranked_ids[:k],
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    precision_at_k=metrics[f"precision@{k}"],
                    recall_at_k=metrics[f"recall@{k}"],
                    mrr_at_k=metrics[f"mrr@{k}"],
                    ndcg_at_k=metrics[f"ndcg@{k}"],
                    latency_ms=latency_ms,
                )
            )
        count = len(results)
        reports[mode] = RetrievalModeReport(
            mode=mode,
            precision_at_k=sum(result.precision_at_k for result in results) / count,
            recall_at_k=sum(result.recall_at_k for result in results) / count,
            mrr_at_k=sum(result.mrr_at_k for result in results) / count,
            ndcg_at_k=sum(result.ndcg_at_k for result in results) / count,
            mean_latency_ms=sum(result.latency_ms for result in results) / count,
            cases=results,
        )

    return RetrievalEvalReport(dataset=dataset.name, k=k, modes=reports)
