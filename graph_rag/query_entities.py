import difflib
import re

import networkx as nx

import config


def _all_name_variants(graph: nx.Graph) -> dict[str, str]:
    """Maps lowercased name/alias -> canonical node name."""
    variants: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        variants[node.lower()] = node
        for alias in data.get("aliases", []):
            variants.setdefault(alias.lower(), node)
    return variants


def _question_ngrams(question: str, max_n: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", question)
    ngrams = []
    for n in range(max_n, 0, -1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def find_entities_in_question(question: str, graph: nx.Graph) -> list[str]:
    """Canonical entity names mentioned in the question: substring match
    against known names/aliases first (HotpotQA questions usually name
    entities verbatim), then difflib fuzzy match on question n-grams for
    near-misses. No LLM call.
    """
    variants = _all_name_variants(graph)
    q_lower = question.lower()

    matched: set[str] = set()
    for variant, canonical in variants.items():
        if len(variant) >= 3 and variant in q_lower:
            matched.add(canonical)

    if not matched:
        # difflib ratios are unstable on very short strings (e.g. "is" vs
        # "iOS" scores 0.8+ purely from sharing 2 letters) -- restrict fuzzy
        # matching to ngrams and candidate names of reasonable length so it
        # only catches genuine near-misses, not short-word noise.
        fuzzy_candidates = [v for v in variants if len(v) >= 4]
        for ngram in _question_ngrams(question):
            if len(ngram) < 4:
                continue
            close = difflib.get_close_matches(
                ngram.lower(), fuzzy_candidates, n=1, cutoff=config.ENTITY_FUZZY_MATCH_THRESHOLD
            )
            for c in close:
                matched.add(variants[c])

    return sorted(matched)
