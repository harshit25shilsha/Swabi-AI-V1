"""Compare dense, BM25, and hybrid RAG retrieval against labeled queries.

python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --dataset app/evaluation/datasets/retrieval_v1.json --k 5
python scripts/evaluate_retrieval.py --output retrieval-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.dataset import EvalDataset  # noqa: E402
from app.evaluation.retrieval import (  # noqa: E402
    RetrievalEvalDataset,
    evaluate_retrieval,
)
from app.rag.embeddings import EmbeddingError, EmbeddingModel  # noqa: E402
from app.rag.qdrant_store import QdrantStore, QdrantStoreError  # noqa: E402
from app.rag.retriever import RagRetriever, RetrieverError  # noqa: E402
from app.rag.version_check import check_version_authority  # noqa: E402

DEFAULT_DATASET = Path("app/evaluation/datasets/retrieval_v1.json")


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, and hybrid retrieval against labeled queries."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--rrf-rank-constant",
        type=int,
        default=60,
        help="RRF rank constant to evaluate (default: 60)",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=4,
        help="Dense/BM25 candidate depth as a multiple of k (default: 4)",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _load_dataset(path: Path) -> RetrievalEvalDataset:
    dataset = EvalDataset.load(path)
    payload: dict[str, Any] = {
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
    return RetrievalEvalDataset.model_validate(payload)


def _searchers(
    index_version: str,
    *,
    rrf_rank_constant: int,
    candidate_multiplier: int,
):
    embedder = EmbeddingModel()
    embedder.verify_ready()
    store = QdrantStore()
    hybrid = RagRetriever(embedder=embedder, store=store)
    min_score = settings.rag_min_score

    def dense(case, limit):
        vectors = embedder.embed_queries([case.query])
        if not len(vectors):
            return []
        return store.search(
            vectors[0].tolist(),
            top_k=limit,
            audience=case.audience,
            area=case.area,
            min_score=min_score,
            index_version=index_version,
        )

    def keyword(case, limit):
        return store.keyword_search(
            case.query,
            top_k=limit,
            audience=case.audience,
            area=case.area,
            index_version=index_version,
        )

    def hybrid_search(case, limit):
        return hybrid.search(
            case.query,
            audience=case.audience,
            area=case.area,
            top_k=limit,
            min_score=min_score,
            index_version=index_version,
            rrf_rank_constant=rrf_rank_constant,
            candidate_multiplier=candidate_multiplier,
        )

    return {
        "dense": dense,
        "bm25": keyword,
        "hybrid": hybrid_search,
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _args(argv)
    if args.k < 1:
        print("[error] --k must be at least 1", file=sys.stderr)
        return 2
    if args.rrf_rank_constant < 1:
        print("[error] --rrf-rank-constant must be at least 1", file=sys.stderr)
        return 2
    if args.candidate_multiplier < 1:
        print("[error] --candidate-multiplier must be at least 1", file=sys.stderr)
        return 2

    dataset = _load_dataset(args.dataset)
    version = check_version_authority()
    if not version.ok:
        print(
            f"[error] knowledge index is not authoritative: {version.reason}",
            file=sys.stderr,
        )
        return 1

    try:
        report = evaluate_retrieval(
            dataset,
            _searchers(
                version.local_version,
                rrf_rank_constant=args.rrf_rank_constant,
                candidate_multiplier=args.candidate_multiplier,
            ),
            k=args.k,
        )
    except (EmbeddingError, QdrantStoreError, RetrieverError) as exc:
        print(f"[error] retrieval evaluation failed: {exc}", file=sys.stderr)
        return 1

    output = report.to_dict()
    output["knowledge_version"] = version.local_version
    output["embedding_model"] = settings.rag_embedding_model
    output["dense_min_score"] = settings.rag_min_score
    output["rrf_rank_constant"] = args.rrf_rank_constant
    output["candidate_multiplier"] = args.candidate_multiplier
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[ok] wrote retrieval report to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
