import json

import networkx as nx

import config
from common.llm_client import generate

SUMMARY_PROMPT_TEMPLATE = """The following entities and relationships form a cluster in a knowledge graph. \
Write a short (2-4 sentence) plain-text summary of what this cluster is about.

Entities: {entities}

Relationships:
{relations}

Summary:"""


def _format_relations(graph: nx.Graph, nodes: list[str]) -> str:
    node_set = set(nodes)
    seen = set()
    lines = []
    for a, b, data in graph.edges(nodes, data=True):
        if a not in node_set or b not in node_set:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        relation_label = data["provenance"][0]["relation"] if data.get("provenance") else "related to"
        lines.append(f"- {a} {relation_label} {b}")
    return "\n".join(lines) if lines else "(no explicit relations)"


def summarize_community(graph: nx.Graph, nodes: list[str]) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        entities=", ".join(nodes),
        relations=_format_relations(graph, nodes),
    )
    return generate(prompt, pipeline="graph_rag", stage="summarization").strip()


def summarize_communities(
    graph: nx.Graph,
    members: dict[int, list[str]],
    min_size: int = config.MIN_COMMUNITY_SIZE_FOR_SUMMARY,
) -> dict[int, str]:
    """Communities below min_size get a cheap templated description instead
    of an LLM call -- a 1-2 node cluster has nothing to meaningfully compress.
    """
    summaries: dict[int, str] = {}
    for community_id, nodes in members.items():
        if len(nodes) < min_size:
            summaries[community_id] = f"Small cluster: {', '.join(nodes)}."
            continue
        summaries[community_id] = summarize_community(graph, nodes)
        print(f"  summarized community {community_id} ({len(nodes)} entities)")
    return summaries


def save_summaries(summaries: dict[int, str], path=config.COMMUNITY_SUMMARIES_PATH) -> None:
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def load_summaries(path=config.COMMUNITY_SUMMARIES_PATH) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}
