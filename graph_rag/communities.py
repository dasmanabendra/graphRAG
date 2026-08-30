import json

import networkx as nx
from networkx.algorithms.community import louvain_communities

import config


def detect_communities(graph: nx.Graph, resolution: float = config.LOUVAIN_RESOLUTION) -> dict[str, int]:
    """Returns {node_name: community_id}."""
    node_sets = louvain_communities(graph, weight="weight", resolution=resolution, seed=42)
    assignment: dict[str, int] = {}
    for community_id, nodes in enumerate(node_sets):
        for node in nodes:
            assignment[node] = community_id
    return assignment


def community_members(assignment: dict[str, int]) -> dict[int, list[str]]:
    """Sorted so summarization prompts are deterministic across process
    runs -- louvain_communities returns sets internally, and Python's
    string hash randomization makes set iteration order vary run to run,
    which otherwise causes spurious response-cache misses.
    """
    members: dict[int, list[str]] = {}
    for node, community_id in assignment.items():
        members.setdefault(community_id, []).append(node)
    return {community_id: sorted(nodes) for community_id, nodes in members.items()}


def save_communities(assignment: dict[str, int], path=config.COMMUNITIES_PATH) -> None:
    path.write_text(json.dumps(assignment, indent=2), encoding="utf-8")


def load_communities(path=config.COMMUNITIES_PATH) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))
