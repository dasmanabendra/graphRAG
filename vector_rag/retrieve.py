import config
from common.embeddings import embed_text
from vector_rag.index import get_collection


def retrieve(question: str, k: int = config.VECTOR_TOP_K) -> list[dict]:
    query_embedding = embed_text(question, pipeline="vector_rag", stage="query", task_type="RETRIEVAL_QUERY")
    collection = get_collection()
    result = collection.query(query_embeddings=[query_embedding], n_results=k)

    chunks = []
    for cid, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({"chunk_id": cid, "text": doc, "title": meta["title"], "distance": dist})
    return chunks
