"""Tests for ranking metrics and retrieval evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.evaluation.retrieval import (
    RetrievalEvalCase,
    RetrievalEvalDataset,
    evaluate_retrieval,
    ranking_metrics,
)
from app.rag.schemas import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id="1.1",
        section_path="Policies",
        text="Text",
        score=0.0,
    )


def test_ranking_metrics_calculates_precision_recall_mrr_and_ndcg():
    metrics = ranking_metrics(
        ["relevant-1", "relevant-2"],
        [
            "irrelevant",
            "relevant-1",
            "relevant-2",
        ],
        k=3,
    )

    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert metrics["ndcg@3"] == pytest.approx(
        (1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156)
    )


@pytest.mark.parametrize("k", [0, -1])
def test_ranking_metrics_rejects_nonpositive_k(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        ranking_metrics(["id"], ["id"], k=k)


def test_ranking_metrics_rejects_empty_relevance_labels():
    with pytest.raises(ValueError, match="at least one relevant"):
        ranking_metrics([], ["id"], k=5)


def test_ranking_metrics_deduplicates_result_ids():
    metrics = ranking_metrics(
        ["relevant"],
        [
            "irrelevant",
            "relevant",
            "relevant",
        ],
        k=3,
    )

    assert metrics["precision@3"] == pytest.approx(1 / 3)
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5


def test_evaluate_retrieval_reports_case_and_mean_metrics():
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "test",
            "cases": [
                {
                    "id": "case-1",
                    "query": "policy query",
                    "relevant_chunk_ids": ["match"],
                },
                {
                    "id": "case-2",
                    "query": "another query",
                    "relevant_chunk_ids": ["missing"],
                },
            ],
        }
    )

    def fake_search(case: RetrievalEvalCase, k: int) -> list[RetrievedChunk]:
        assert case.id in {"case-1", "case-2"}
        assert k == 1
        return [_chunk("match")]

    report = evaluate_retrieval(dataset, {"fake": fake_search}, k=1)

    mode = report.modes["fake"]
    assert mode.precision_at_k == 0.5
    assert mode.recall_at_k == 0.5
    assert mode.mrr_at_k == 0.5
    assert mode.ndcg_at_k == 0.5
    assert len(mode.cases) == 2
    assert mode.cases[0].precision_at_k == 1.0
    assert mode.cases[1].precision_at_k == 0.0
    serialized = report.to_dict()
    assert serialized["dataset"] == "test"
    assert serialized["modes"]["fake"]["precision_at_k"] == 0.5
    assert serialized["modes"]["fake"]["cases"][0]["precision_at_k"] == 1.0


def test_dataset_labels_relevant_chunk_ids():
    case = RetrievalEvalDataset.model_validate(
        {
            "name": "test",
            "cases": [
                {
                    "id": "case",
                    "query": "policy query",
                    "relevant_chunk_ids": ["chunk-1"],
                    "audience": "vendor",
                    "area": "swabi",
                }
            ],
        }
    ).cases[0]

    assert case.relevant_chunk_ids == ["chunk-1"]
    assert case.audience == "vendor"
    assert case.area == "swabi"


def test_checked_in_retrieval_dataset_loads():
    from scripts.evaluate_retrieval import DEFAULT_DATASET, _load_dataset

    dataset = _load_dataset(DEFAULT_DATASET)

    assert len(dataset.cases) == 15
    assert all(case.relevant_chunk_ids for case in dataset.cases)
    cases = {case.id: case for case in dataset.cases}
    assert set(cases["vendor-payouts"].relevant_chunk_ids) == {
        "3b9cd5e35fca7726",
        "dfc03cb4a7b33ef9",
        "deedb842b9015315",
        "00ff3e8b795c4478",
        "863d145f20890b30",
    }
    assert cases["vendor-withdrawal-steps"].relevant_chunk_ids == ["3b9cd5e35fca7726"]
    assert cases["package-booking-through-enquiry"].relevant_chunk_ids == [
        "47c29ba71be7e11c",
        "c3a5eb066f5f711e",
    ]


def test_evaluation_cli_exposes_tuning_options():
    from scripts.evaluate_retrieval import _args

    args = _args(
        [
            "--rrf-rank-constant",
            "20",
            "--candidate-multiplier",
            "2",
            "--reranker-model",
            "local/test-model",
            "--include-reranked",
        ]
    )

    assert args.rrf_rank_constant == 20
    assert args.candidate_multiplier == 2
    assert args.reranker_model == "local/test-model"
    assert args.include_reranked is True


def test_searchers_apply_tuning_only_to_hybrid_search():
    from scripts.evaluate_retrieval import _searchers

    embedder = MagicMock()
    embedder.embed_queries.return_value = np.ones((1, 2), dtype="float32")
    store = MagicMock()
    hybrid = MagicMock()
    reranked = MagicMock()
    case = RetrievalEvalCase(
        id="case",
        query="withdraw payout",
        relevant_chunk_ids=["chunk"],
        audience="vendor",
        area="swabi",
    )

    with (
        patch("scripts.evaluate_retrieval.EmbeddingModel", return_value=embedder),
        patch("scripts.evaluate_retrieval.QdrantStore", return_value=store),
        patch(
            "scripts.evaluate_retrieval.RagRetriever",
            side_effect=[hybrid, reranked],
        ),
        patch("scripts.evaluate_retrieval.CrossEncoderReranker") as reranker,
    ):
        searchers = _searchers(
            "version-1",
            rrf_rank_constant=20,
            candidate_multiplier=2,
            reranker_model="local/test-model",
            include_reranked=True,
        )
        searchers["dense"](case, 5)
        searchers["bm25"](case, 5)
        searchers["hybrid"](case, 5)
        searchers["hybrid_reranked"](case, 5)

    assert store.search.call_args.kwargs["index_version"] == "version-1"
    assert "rrf_rank_constant" not in store.search.call_args.kwargs
    assert "candidate_multiplier" not in store.search.call_args.kwargs
    assert store.keyword_search.call_args.kwargs["index_version"] == "version-1"
    assert hybrid.search.call_args.kwargs["rrf_rank_constant"] == 20
    assert hybrid.search.call_args.kwargs["candidate_multiplier"] == 2
    assert reranker.call_args.args == ("local/test-model",)
    assert reranked.search.call_args.kwargs["index_version"] == "version-1"
    assert reranked.search.call_args.kwargs["rrf_rank_constant"] == 20


def test_retrieval_report_reports_reranking_deltas():
    dataset = RetrievalEvalDataset(
        name="test",
        cases=[
            RetrievalEvalCase(
                id="case",
                query="query",
                relevant_chunk_ids=["match"],
            )
        ],
    )
    report = evaluate_retrieval(
        dataset,
        {
            "hybrid": lambda case, k: [_chunk("miss")],
            "hybrid_reranked": lambda case, k: [_chunk("match")],
        },
        k=1,
    )

    comparison = report.to_dict()["comparisons"]["hybrid_reranked_vs_hybrid"]

    assert comparison["mrr_at_k"] == 1.0
    assert comparison["ndcg_at_k"] == 1.0
