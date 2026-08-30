from common.answer_parsing import parse_reasoned_answer
from common.llm_client import generate

ANSWER_PROMPT_TEMPLATE = """Answer the question using ONLY the passages and graph community summary below.

Respond in exactly this format:
Reasoning: <1-2 sentences on which passage(s) or graph summary support the answer and how>
Answer: <concise final answer -- as few words as possible (a name, date, yes/no, etc.), \
matching the style of the question>

Passages:
{passages}

Graph community summary:
{community_summary}

Question: {question}"""


def answer(question: str, passages: list[dict], community_summaries: list[str]) -> dict:
    passages_text = "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)
    summary_text = "\n".join(community_summaries) if community_summaries else "(none)"
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        passages=passages_text, community_summary=summary_text, question=question
    )
    raw = generate(prompt, pipeline="hybrid_rag", stage="answer").strip()
    return parse_reasoned_answer(raw)
