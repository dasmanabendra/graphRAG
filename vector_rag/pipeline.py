import config
from vector_rag.answer import ANSWER_PROMPT_TEMPLATE
from vector_rag.answer import answer as generate_answer
from vector_rag.index import build_index
from vector_rag.retrieve import retrieve


def run_index() -> None:
    build_index()


def answer_question(question: str) -> dict:
    chunks = retrieve(question)

    passages = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(passages=passages, question=question)
    result = generate_answer(question, chunks)
    prediction = result["answer"]
    reasoning = result["reasoning"]

    steps = [
        {
            "name": "Embed the question",
            "cost": "1 embedding call",
            "description": (
                "Turns the question into a 768-number vector with the same embedding model used "
                "to index the corpus, so it can be compared against stored chunk vectors."
            ),
            "detail": {
                "model": config.EMBEDDING_MODEL,
                "task_type": "RETRIEVAL_QUERY",
                "dimensions": config.EMBEDDING_DIM,
            },
        },
        {
            "name": f"Similarity search (top-{config.VECTOR_TOP_K})",
            "cost": "free",
            "description": (
                "Compares the question's vector to every stored chunk vector and returns the "
                "k closest by distance -- pure math, no LLM call, and no understanding of meaning "
                "beyond what the embedding model already captured."
            ),
            "detail": {
                "results": [
                    {"chunk_id": c["chunk_id"], "title": c["title"], "distance": c["distance"]}
                    for c in chunks
                ],
            },
        },
        {
            "name": "Generate answer",
            "cost": "1 chat call",
            "description": (
                "Sends the retrieved passages and the question to the LLM, instructed to answer "
                "using ONLY those passages and to give its reasoning before the final answer."
            ),
            "detail": {
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": prediction,
            },
        },
    ]

    return {
        "prediction": prediction,
        "reasoning": reasoning,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "steps": steps,
    }
