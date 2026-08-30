from common.answer_parsing import parse_reasoned_answer
from common.llm_client import generate

ANSWER_PROMPT_TEMPLATE = """Answer the question using ONLY the passages below.

Respond in exactly this format:
Reasoning: <1-2 sentences on which passage(s) support the answer and how>
Answer: <concise final answer -- as few words as possible (a name, date, yes/no, etc.), \
matching the style of the question>

Passages:
{passages}

Question: {question}"""


def answer(question: str, chunks: list[dict]) -> dict:
    passages = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(passages=passages, question=question)
    raw = generate(prompt, pipeline="vector_rag", stage="answer").strip()
    return parse_reasoned_answer(raw)
