from typing import Optional

import networkx as nx

import config
from data_prep.build_corpus import load_cached_corpus
from graph_rag.communities import load_communities
from graph_rag.summarization import load_summaries

_corpus_cache: Optional[dict] = None
_communities_cache: Optional[dict] = None
_summaries_cache: Optional[dict] = None


def _get_corpus() -> dict:
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = load_cached_corpus()
    return _corpus_cache


def _get_communities() -> dict:
    global _communities_cache
    if _communities_cache is None:
        _communities_cache = load_communities()
    return _communities_cache


def _get_summaries() -> dict:
    global _summaries_cache
    if _summaries_cache is None:
        _summaries_cache = load_summaries()
    return _summaries_cache


def _neighbors_with_edges(graph: nx.Graph, node: str) -> list[dict]:
    results = []
    for neighbor in graph.neighbors(node):
        data = graph[node][neighbor]
        relation_label = data["provenance"][0]["relation"] if data.get("provenance") else "related to"
        results.append({"neighbor": neighbor, "relation": relation_label})
    return results


def find_bridge_nodes(graph: nx.Graph, matched_entities: list[str]) -> list[str]:
    """Neighbors shared by two or more matched entities -- the connector
    entity for questions that name two entities but require chaining
    through a third, unnamed one (HotpotQA's 'bridge' question type).
    Reported separately for observability; the actual context-gathering
    fix is expand_neighbors below, which pulls in ALL 1-hop neighbors
    (not just shared ones) since a single-entity bridge question (e.g.
    "what position did the woman who starred in X hold") needs the
    unmatched neighbor's own source passage, not just its name.
    """
    bridges: set[str] = set()
    for i in range(len(matched_entities)):
        for j in range(i + 1, len(matched_entities)):
            a, b = matched_entities[i], matched_entities[j]
            if a not in graph or b not in graph or graph.has_edge(a, b):
                continue
            bridges |= set(graph.neighbors(a)) & set(graph.neighbors(b))
    return sorted(bridges)


def expand_neighbors(graph: nx.Graph, matched_entities: list[str], cap: int) -> list[str]:
    """1-hop neighbors of every matched entity, highest-weight edges first,
    capped so a high-degree hub node can't blow up the context.
    """
    weighted: dict[str, int] = {}
    for entity in matched_entities:
        if entity not in graph:
            continue
        for neighbor in graph.neighbors(entity):
            weight = graph[entity][neighbor].get("weight", 1)
            weighted[neighbor] = max(weighted.get(neighbor, 0), weight)

    ranked = sorted(weighted, key=lambda n: -weighted[n])
    return ranked[:cap]


def assemble_context(question: str, graph: nx.Graph, matched_entities: list[str]) -> dict:
    corpus = _get_corpus()
    communities = _get_communities()
    summaries = _get_summaries()

    bridge_nodes = find_bridge_nodes(graph, matched_entities)
    neighbor_entities = expand_neighbors(graph, matched_entities, cap=config.MAX_LOCAL_SEARCH_NEIGHBORS)
    focus_entities = list(dict.fromkeys(matched_entities + neighbor_entities))

    entity_lines: list[str] = []
    relations: list[dict] = []
    chunk_ids: set[str] = set()
    chunk_sources: dict[str, list[str]] = {}
    community_ids: set[int] = set()

    matched_set = set(matched_entities)
    for entity in focus_entities:
        if entity not in graph:
            continue
        data = graph.nodes[entity]
        entity_lines.append(f"- {entity} ({data.get('type', 'unknown')})")
        for cid in data.get("chunk_ids", []):
            chunk_ids.add(cid)
            chunk_sources.setdefault(cid, []).append(entity)

        community_id = communities.get(entity)
        if community_id is not None:
            community_ids.add(community_id)

        # Only expand relation lines from entities actually named in the
        # question -- their neighbors' own chunk text is still pulled in
        # above, but listing THEIR relations too would add a 2nd hop of
        # noise without a matching 2nd hop of retrieved passages to ground it.
        if entity in matched_set:
            for nb in _neighbors_with_edges(graph, entity):
                entity_lines.append(f"    -> {nb['relation']} -> {nb['neighbor']}")
                relations.append({"from": entity, "relation": nb["relation"], "to": nb["neighbor"]})

    passages = []
    for cid in sorted(chunk_ids):
        chunk = corpus.get(cid)
        if chunk:
            passages.append(f"[{chunk['title']}] {chunk['text']}")

    community_texts = [summaries[cid] for cid in sorted(community_ids) if cid in summaries]

    context_parts = []
    if entity_lines:
        context_parts.append("Entities & relations:\n" + "\n".join(entity_lines))
    if passages:
        context_parts.append("Source passages:\n" + "\n\n".join(passages))
    if community_texts:
        context_parts.append("Community summary:\n" + "\n".join(community_texts))

    return {
        "context": "\n\n".join(context_parts) if context_parts else "(no relevant graph context found)",
        "chunk_ids": sorted(chunk_ids),
        "chunk_sources": chunk_sources,
        "matched_entities": matched_entities,
        "neighbor_entities": neighbor_entities,
        "bridge_nodes": bridge_nodes,
        "relations": relations,
        "community_ids": sorted(community_ids),
        "community_summaries": community_texts,
    }
