from typing import Optional

import chromadb

import config
from common.embeddings import embed_texts
from data_prep.build_corpus import load_cached_corpus


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(name="corpus", embedding_function=None)


def build_index(corpus: Optional[dict[str, dict]] = None) -> None:
    if corpus is None:
        corpus = load_cached_corpus()

    chunk_ids = list(corpus.keys())
    texts = [corpus[cid]["text"] for cid in chunk_ids]
    titles = [corpus[cid]["title"] for cid in chunk_ids]

    embeddings = embed_texts(texts, pipeline="vector_rag", stage="index", task_type="RETRIEVAL_DOCUMENT")

    collection = get_collection()
    # upsert (not add) so re-running during dev iteration doesn't error on
    # duplicate ids -- cached embeddings make this free after the first pass.
    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": t} for t in titles],
    )
    print(f"Indexed {len(chunk_ids)} chunks into Chroma collection 'corpus' (count={collection.count()}).")
