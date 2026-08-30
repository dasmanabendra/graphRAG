RRF_K = 60


def reciprocal_rank_fusion(
    vector_chunk_ids: list[str],
    graph_chunk_ids: list[str],
    top_k: int,
) -> list[dict]:
    """Merge two independently-ranked chunk_id lists into one ranking.

    Each list contributes 1/(RRF_K + rank) per chunk_id it contains (1-based
    rank). A chunk present in both lists gets both contributions, so chunks
    that are both vector-similar AND graph-connected naturally float to the
    top -- no reranker model or extra LLM call needed.
    """
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for rank, cid in enumerate(vector_chunk_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
        sources.setdefault(cid, set()).add("vector")

    for rank, cid in enumerate(graph_chunk_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
        sources.setdefault(cid, set()).add("graph")

    ranked = sorted(scores, key=lambda cid: -scores[cid])[:top_k]
    return [
        {"chunk_id": cid, "score": scores[cid], "found_via": sorted(sources[cid])}
        for cid in ranked
    ]
