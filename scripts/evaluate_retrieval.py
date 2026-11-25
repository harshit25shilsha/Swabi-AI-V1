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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.capabilities.support.retrieval_query import (  # noqa: E402
    prepare_retrieval_query,
)
from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.retrieval import (  # noqa: E402
    RetrievalEvalDataset,
    evaluate_retrieval,
    load_retrieval_dataset,
)
from app.rag.embeddings import EmbeddingError, EmbeddingModel  # noqa: E402
from app.rag.qdrant_store import QdrantStore, QdrantStoreError  # noqa: E402
from app.rag.reranker import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    RerankerError,
)
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
    parser.add_argument(
        "--reranker-model",
        default=settings.rag_reranker_model or DEFAULT_CROSS_ENCODER_MODEL,
        help="Local cross-encoder model used by the hybrid_reranked mode",
    )
    parser.add_argument(
        "--include-reranked",
        action="store_true",
        help="Evaluate the optional local cross-encoder reranker",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _load_dataset(path: Path) -> RetrievalEvalDataset:
    return load_retrieval_dataset(path)


def _searchers(
    index_version: str,
    *,
    rrf_rank_constant: int,
    candidate_multiplier: int,
    reranker_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    include_reranked: bool = False,
):
    embedder = EmbeddingModel()
    embedder.verify_ready()
    store = QdrantStore()
    hybrid = RagRetriever(embedder=embedder, store=store)
    min_score = settings.rag_min_score

    def dense(case, limit):
        query = prepare_retrieval_query(case.query)
        vectors = embedder.embed_queries([query])
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
            prepare_retrieval_query(case.query),
            top_k=limit,
            audience=case.audience,
            area=case.area,
            index_version=index_version,
        )

    def hybrid_search(case, limit):
        return hybrid.search(
            prepare_retrieval_query(case.query),
            audience=case.audience,
            area=case.area,
            top_k=limit,
            min_score=min_score,
            index_version=index_version,
            rrf_rank_constant=rrf_rank_constant,
            candidate_multiplier=candidate_multiplier,
        )

    searchers = {
        "dense": dense,
        "bm25": keyword,
        "hybrid": hybrid_search,
    }
    if include_reranked:
        hybrid_reranked = RagRetriever(
            embedder=embedder,
            store=store,
            reranker=CrossEncoderReranker(reranker_model),
        )

        def hybrid_reranked_search(case, limit):
            return hybrid_reranked.search(
                case.query,
                audience=case.audience,
                area=case.area,
                top_k=limit,
                min_score=min_score,
                index_version=index_version,
                rrf_rank_constant=rrf_rank_constant,
                candidate_multiplier=candidate_multiplier,
            )

        searchers["hybrid_reranked"] = hybrid_reranked_search
    return searchers


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
                reranker_model=args.reranker_model,
                include_reranked=args.include_reranked,
            ),
            k=args.k,
        )
    except (EmbeddingError, QdrantStoreError, RerankerError, RetrieverError) as exc:
        print(f"[error] retrieval evaluation failed: {exc}", file=sys.stderr)
        return 1

    output = report.to_dict()
    output["knowledge_version"] = version.local_version
    output["embedding_model"] = settings.rag_embedding_model
    output["dense_min_score"] = settings.rag_min_score
    output["rrf_rank_constant"] = args.rrf_rank_constant
    output["candidate_multiplier"] = args.candidate_multiplier
    output["reranker_enabled"] = args.include_reranked
    if args.include_reranked:
        output["reranker_model"] = args.reranker_model
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[ok] wrote retrieval report to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
