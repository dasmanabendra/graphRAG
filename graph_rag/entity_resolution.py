import difflib
from typing import Optional

import config
from graph_rag.extraction import ExtractedEntity


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


class EntityResolver:
    """Merges duplicate entity surface forms with pure-Python string
    matching (exact, then difflib fuzzy) -- no extra LLM call. Entities only
    merge when they share a (case-insensitive) type: extraction sometimes
    yields same-name entities of different types (e.g. the film "Ed Wood"
    vs. the person "Ed Wood"), which must stay distinct nodes.
    """

    def __init__(self):
        self.canonical: dict[str, dict] = {}  # canonical_name -> {name, type, aliases, chunk_ids}
        self._norm_to_canonical: dict[str, str] = {}  # "type::normalized_name" -> canonical_name

    def _fuzzy_match(self, etype: str, norm: str) -> Optional[str]:
        best_ratio = 0.0
        best_canonical = None
        for key, canonical_name in self._norm_to_canonical.items():
            key_type, key_norm = key.split("::", 1)
            if key_type != etype:
                continue
            ratio = difflib.SequenceMatcher(None, norm, key_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_canonical = canonical_name
        return best_canonical if best_ratio >= config.ENTITY_FUZZY_MATCH_THRESHOLD else None

    def add(self, entity: ExtractedEntity) -> str:
        etype = entity.type.strip().lower()
        norm = _normalize(entity.name)
        lookup_key = f"{etype}::{norm}"

        canonical_name = self._norm_to_canonical.get(lookup_key) or self._fuzzy_match(etype, norm)
        if canonical_name is None:
            canonical_name = entity.name
            self.canonical[canonical_name] = {
                "name": canonical_name,
                "type": etype,
                "aliases": set(),
                "chunk_ids": set(),
            }

        record = self.canonical[canonical_name]
        record["aliases"].add(entity.name)
        record["chunk_ids"].add(entity.chunk_id)
        self._norm_to_canonical[lookup_key] = canonical_name
        return canonical_name

    def resolve_name(self, name: str, type_hint: Optional[str] = None) -> Optional[str]:
        """Best-effort canonical lookup for a bare name (e.g. a relation
        source/target) that may not have been seen via add() directly.
        """
        norm = _normalize(name)
        if type_hint:
            exact = self._norm_to_canonical.get(f"{type_hint.strip().lower()}::{norm}")
            if exact:
                return exact
            fuzzy = self._fuzzy_match(type_hint.strip().lower(), norm)
            if fuzzy:
                return fuzzy

        # No type hint, or no match within that type: search all types.
        for key, canonical_name in self._norm_to_canonical.items():
            if key.split("::", 1)[1] == norm:
                return canonical_name

        best_ratio, best_canonical = 0.0, None
        for key, canonical_name in self._norm_to_canonical.items():
            ratio = difflib.SequenceMatcher(None, norm, key.split("::", 1)[1]).ratio()
            if ratio > best_ratio:
                best_ratio, best_canonical = ratio, canonical_name
        return best_canonical if best_ratio >= config.ENTITY_FUZZY_MATCH_THRESHOLD else None

    def resolved_entities(self) -> dict[str, dict]:
        return {
            name: {
                "name": name,
                "type": record["type"],
                "aliases": sorted(record["aliases"]),
                "chunk_ids": sorted(record["chunk_ids"]),
            }
            for name, record in self.canonical.items()
        }


def resolve_entities(entities: list[ExtractedEntity]) -> tuple[EntityResolver, dict[str, dict]]:
    resolver = EntityResolver()
    for entity in entities:
        resolver.add(entity)
    return resolver, resolver.resolved_entities()
