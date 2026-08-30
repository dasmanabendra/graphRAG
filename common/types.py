from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class Entity:
    name: str
    type: str
    chunk_ids: list[str] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    chunk_id: str


@dataclass
class Community:
    community_id: int
    entity_names: list[str]
    summary: str = ""


@dataclass
class EvalRecord:
    question_id: str
    question: str
    gold_answer: str
    gold_titles: list[str]
    vector_prediction: str = ""
    vector_chunk_ids: list[str] = field(default_factory=list)
    vector_calls: int = 0
    graph_prediction: str = ""
    graph_chunk_ids: list[str] = field(default_factory=list)
    graph_calls: int = 0
