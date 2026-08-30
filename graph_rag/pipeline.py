from typing import Optional

from data_prep.build_corpus import load_cached_corpus
from graph_rag.answer import ANSWER_PROMPT_TEMPLATE
from graph_rag.answer import answer as generate_answer
from graph_rag.communities import community_members, detect_communities, save_communities
from graph_rag.entity_resolution import resolve_entities
from graph_rag.extraction import extract_corpus
from graph_rag.graph_build import build_graph, load_graph, save_graph
from graph_rag.local_search import assemble_context
from graph_rag.query_entities import find_entities_in_question
from graph_rag.summarization import save_summaries, summarize_communities

_graph_cache = None


def get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = load_graph()
    return _graph_cache


def run_index(corpus: Optional[dict[str, dict]] = None) -> None:
    if corpus is None:
        corpus = load_cached_corpus()

    print("Extracting entities/relations...")
    extraction = extract_corpus(corpus)

    print("Resolving entities...")
    resolver, _ = resolve_entities(extraction.entities)

    print("Building graph...")
    graph = build_graph(extraction, resolver)
    save_graph(graph)

    print("Detecting communities...")
    assignment = detect_communities(graph)
    members = community_members(assignment)
    save_communities(assignment)

    print("Summarizing communities...")
    summaries = summarize_communities(graph, members)
    save_summaries(summaries)

    global _graph_cache
    _graph_cache = graph

    print(
        f"GraphRAG index built: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {len(members)} communities."
    )


def answer_question(question: str) -> dict:
    graph = get_graph()

    matched = find_entities_in_question(question, graph)

    ctx = assemble_context(question, graph, matched)
    focus_count = len(set(matched) | set(ctx["neighbor_entities"]))

    prompt = ANSWER_PROMPT_TEMPLATE.format(context=ctx["context"], question=question)
    result = generate_answer(question, ctx["context"])
    prediction = result["answer"]
    reasoning = result["reasoning"]

    steps = [
        {
            "name": "Match entities in question",
            "cost": "free",
            "description": (
                "Checks the question's text against every known entity name/alias in the graph -- "
                "exact substring match first, fuzzy match as a fallback. No LLM, no embeddings, just "
                "string matching; works because HotpotQA questions usually name their subject directly."
            ),
            "detail": {
                "matched_entities": matched,
            },
        },
        {
            "name": "Expand graph neighbors (1-hop)",
            "cost": "free",
            "description": (
                "Walks one edge out from each matched entity to pull in connected entities, ranked "
                "by edge weight and capped at 20. This -- not the 'bridge nodes' label -- is what "
                "actually solves bridge-type questions, by reaching entities the question never named. "
                "'Bridge nodes' just highlights, for display only, which of these neighbors connects "
                "two matched entities at once; it's a subset of what's already gathered here."
            ),
            "detail": {
                "summary": (
                    f"{len(matched)} matched -> {len(ctx['neighbor_entities'])} neighbors "
                    f"-> {focus_count} focus entities"
                ),
                "neighbor_entities": ctx["neighbor_entities"],
                "bridge_nodes": ctx["bridge_nodes"],
                "relations": ctx["relations"],
            },
        },
        {
            "name": "Gather source passages & community summary",
            "cost": "free",
            "description": (
                "For every matched + neighbor entity, pulls the original chunk text it was extracted "
                "from, plus any community summary its cluster got during indexing -- together this "
                "becomes the context sent to the LLM. Can include chunks from unrelated questions if "
                "a shared hub entity (e.g. a city name) links to them."
            ),
            "detail": {
                "chunk_ids": ctx["chunk_ids"],
                "chunk_sources": ctx["chunk_sources"],
                "community_ids": ctx["community_ids"],
                "community_summaries": ctx["community_summaries"],
            },
        },
        {
            "name": "Generate answer",
            "cost": "1 chat call",
            "description": (
                "Sends the assembled entities/relations, source passages, and community summaries to "
                "the LLM, instructed to answer using ONLY that context and to give its reasoning "
                "before the final answer."
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
        "chunk_ids": ctx["chunk_ids"],
        "matched_entities": matched,
        "steps": steps,
    }
