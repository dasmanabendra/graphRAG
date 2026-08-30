from pydantic import BaseModel

import config
from common.llm_client import generate


class ExtractedEntity(BaseModel):
    chunk_id: str
    name: str
    type: str  # PERSON, ORG, LOCATION, WORK, EVENT, OTHER


class ExtractedRelation(BaseModel):
    chunk_id: str
    source: str
    relation: str
    target: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


EXTRACTION_PROMPT_TEMPLATE = """You are extracting entities and relationships from short passages to build a knowledge graph.

For EACH passage below, identify the named entities mentioned (people, organizations, locations, \
creative works, events -- skip anything not clearly one of those types) and the relationships between \
entities stated in the text.

Passages:
{passages}

Return entities and relations for every passage, tagging each with its chunk_id so extractions can be \
traced back to source. Keep relation labels short (2-4 words, e.g. "directed by", "born in", "starred in")."""


def _format_passages(chunks: list[dict]) -> str:
    return "\n\n".join(f"[chunk_id={c['chunk_id']}] {c['text']}" for c in chunks)


def extract_batch(chunks: list[dict]) -> ExtractionResult:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(passages=_format_passages(chunks))
    text = generate(prompt, pipeline="graph_rag", stage="extraction", response_schema=ExtractionResult)
    return ExtractionResult.model_validate_json(text)


def extract_corpus(
    corpus: dict[str, dict], batch_size: int = config.EXTRACTION_BATCH_SIZE
) -> ExtractionResult:
    chunk_list = list(corpus.values())
    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []

    for i in range(0, len(chunk_list), batch_size):
        batch = chunk_list[i : i + batch_size]
        result = extract_batch(batch)
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)
        print(f"  extracted batch {i // batch_size + 1}: {len(result.entities)} entities, {len(result.relations)} relations")

    return ExtractionResult(entities=all_entities, relations=all_relations)
