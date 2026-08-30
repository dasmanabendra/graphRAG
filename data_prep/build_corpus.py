import json
from typing import Optional

import config


def build_corpus(raw_examples: Optional[list[dict]] = None) -> dict[str, dict]:
    """Dedupe paragraphs across examples into a title-keyed corpus. HotpotQA
    distractor paragraphs are static per Wikipedia title, so the same title
    appearing in multiple examples' context always carries the same text --
    deduping on title is exact, not approximate.
    """
    if raw_examples is None:
        raw_examples = json.loads(config.RAW_HOTPOTQA_PATH.read_text(encoding="utf-8"))

    corpus: dict[str, dict] = {}
    for example in raw_examples:
        titles = example["context"]["title"]
        sentences_per_para = example["context"]["sentences"]
        for title, sentences in zip(titles, sentences_per_para):
            if title in corpus:
                continue
            corpus[title] = {
                "chunk_id": title,
                "title": title,
                "text": "".join(sentences),
            }

    config.CORPUS_PATH.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"Built corpus: {len(corpus)} unique chunks from {len(raw_examples)} examples.")
    return corpus


def load_cached_corpus() -> dict[str, dict]:
    return json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
