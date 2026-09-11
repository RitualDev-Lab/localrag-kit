"""Reciprocal Rank Fusion (RRF) for blending dense vector and BM25 keyword rankings."""

from typing import Dict, List, Optional
from localrag.storage.sqlite_store import SearchResult


def reciprocal_rank_fusion(
    bm25_results: List[SearchResult],
    vector_results: List[SearchResult],
    k_rrf: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    limit: int = 10,
) -> List[SearchResult]:
    """
    Merge BM25 keyword results and dense vector similarity results using Reciprocal Rank Fusion.
    
    Formula: RRF_score(d) = sum( weight_m / (k_rrf + rank_m(d)) ) for each ranking system m.
    Default k_rrf = 60 (standard in modern information retrieval).
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}
    sources: Dict[str, List[str]] = {}

    # 1. Score BM25 rankings
    for rank, res in enumerate(bm25_results, start=1):
        cid = res.chunk.id
        chunk_map[cid] = res
        sources.setdefault(cid, []).append("bm25")
        rrf_val = bm25_weight / (k_rrf + rank)
        scores[cid] = scores.get(cid, 0.0) + rrf_val

    # 2. Score Vector rankings
    for rank, res in enumerate(vector_results, start=1):
        cid = res.chunk.id
        chunk_map[cid] = res
        sources.setdefault(cid, []).append("vector")
        rrf_val = vector_weight / (k_rrf + rank)
        scores[cid] = scores.get(cid, 0.0) + rrf_val

    # Sort chunks by fused score descending
    sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    merged: List[SearchResult] = []
    for cid in sorted_cids[:limit]:
        original_res = chunk_map[cid]
        match_types = "+".join(sources.get(cid, ["unknown"]))
        merged.append(
            SearchResult(
                chunk=original_res.chunk,
                score=scores[cid],
                match_type=f"hybrid({match_types})",
            )
        )

    return merged
