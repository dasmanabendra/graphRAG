import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_rag.graph_build import load_graph


def main():
    graph = load_graph()
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    type_counts = Counter(data.get("type", "unknown") for _, data in graph.nodes(data=True))
    print("Node types:", dict(type_counts))

    degrees = sorted(graph.degree, key=lambda x: -x[1])[:10]
    print("Top-10 nodes by degree:")
    for name, degree in degrees:
        print(f"  {name}: {degree}")


if __name__ == "__main__":
    main()
