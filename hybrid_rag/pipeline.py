import config
from data_prep.build_corpus import load_cached_corpus
from graph_rag.local_search import assemble_context
from graph_rag.pipeline import get_graph
from graph_rag.query_entities import find_entities_in_question
from hybrid_rag.answer import ANSWER_PROMPT_TEMPLATE
from hybrid_rag.answer import answer as generate_answer
from hybrid_rag.fuse import reciprocal_rank_fusion
from vector_rag.retrieve import retrieve

HYBRID_TOP_K = 6


def answer_question(question: str) -> dict:
    graph = get_graph()
    corpus = load_cached_corpus()

    vector_chunks = retrieve(question)
    vector_chunk_ids = [c["chunk_id"] for c in vector_chunks]
    vector_lookup = {c["chunk_id"]: c for c in vector_chunks}

    matched = find_entities_in_question(question, graph)
    ctx = assemble_context(question, graph, matched)
    graph_chunk_ids = sorted(
        ctx["chunk_ids"], key=lambda cid: -len(ctx["chunk_sources"].get(cid, []))
    )

    fused = reciprocal_rank_fusion(vector_chunk_ids, graph_chunk_ids, top_k=HYBRID_TOP_K)

    passages = []
    for f in fused:
        cid = f["chunk_id"]
        if cid in vector_lookup:
            passages.append({"chunk_id": cid, "title": vector_lookup[cid]["title"], "text": vector_lookup[cid]["text"]})
        else:
            chunk = corpus.get(cid)
            if chunk:
                passages.append({"chunk_id": cid, "title": chunk["title"], "text": chunk["text"]})

    passages_text = "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)
    summary_text = "\n".join(ctx["community_summaries"]) if ctx["community_summaries"] else "(none)"
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        passages=passages_text, community_summary=summary_text, question=question
    )
    result = generate_answer(question, passages, ctx["community_summaries"])
    prediction = result["answer"]
    reasoning = result["reasoning"]

    steps = [
        {
            "name": "Vector retrieval (embed + similarity search)",
            "cost": "1 embedding call",
            "description": (
                "Same as Vector RAG's first two steps: embed the question, then find the "
                f"top-{config.VECTOR_TOP_K} closest chunks by distance. This is the half that catches "
                "semantically-similar chunks even when they share no graph edge with a matched entity."
            ),
            "detail": {
                "model": config.EMBEDDING_MODEL,
                "task_type": "RETRIEVAL_QUERY",
                "dimensions": config.EMBEDDING_DIM,
                "results": [
                    {"chunk_id": c["chunk_id"], "title": c["title"], "distance": c["distance"]}
                    for c in vector_chunks
                ],
            },
        },
        {
            "name": "Graph retrieval (match entities + expand + gather context)",
            "cost": "free",
            "description": (
                "Same as GraphRAG's first three steps: match entities named in the question, walk "
                "1-hop neighbors, and collect the chunk_ids each focus entity points to. This is the "
                "half that catches multi-hop connections vector similarity alone can't reach."
            ),
            "detail": {
                "matched_entities": matched,
                "neighbor_entities": ctx["neighbor_entities"],
                "community_summaries": ctx["community_summaries"],
            },
        },
        {
            "name": f"Fuse vector + graph results (top-{HYBRID_TOP_K})",
            "cost": "free",
            "description": (
                "Reciprocal Rank Fusion: each chunk_id scores 1/(60+rank) per list it appears in, "
                "summed across both lists. A chunk found by both vector AND graph retrieval outscores "
                "one found by only one -- no reranker model, no extra LLM call, pure arithmetic."
            ),
            "detail": {
                "summary": (
                    f"{len(vector_chunk_ids)} from vector, {len(graph_chunk_ids)} from graph "
                    f"-> {len(fused)} chunks after fusion"
                ),
                "fused_results": fused,
            },
        },
        {
            "name": "Generate answer",
            "cost": "1 chat call",
            "description": (
                "Sends the fused passages and the graph's community summary to the LLM, instructed "
                "to answer using ONLY that context and to give its reasoning before the final answer."
            ),
            "detail": {
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": prediction,
            },
        },
    ]

    return {
        "prediction": prediction,
        "reasoning": reasoning,
        "chunk_ids": [p["chunk_id"] for p in passages],
        "matched_entities": matched,
        "steps": steps,
    }
