import pickle

import networkx as nx

import config
from graph_rag.entity_resolution import EntityResolver
from graph_rag.extraction import ExtractionResult


def build_graph(extraction: ExtractionResult, resolver: EntityResolver) -> nx.Graph:
    graph = nx.Graph()

    for name, record in resolver.resolved_entities().items():
        graph.add_node(name, type=record["type"], aliases=record["aliases"], chunk_ids=record["chunk_ids"])

    for relation in extraction.relations:
        source = resolver.resolve_name(relation.source)
        target = resolver.resolve_name(relation.target)
        if source is None or target is None or source == target:
            continue

        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
            graph[source][target]["provenance"].append(
                {"relation": relation.relation, "chunk_id": relation.chunk_id}
            )
        else:
            graph.add_edge(
                source,
                target,
                weight=1,
                provenance=[{"relation": relation.relation, "chunk_id": relation.chunk_id}],
            )

    return graph


def save_graph(graph: nx.Graph, path=config.GRAPH_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path=config.GRAPH_PATH) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)
