from common.answer_parsing import parse_reasoned_answer
from common.llm_client import generate

ANSWER_PROMPT_TEMPLATE = """Answer the question using ONLY the graph context below.

Respond in exactly this format:
Reasoning: <1-2 sentences on which entities/relations/passages support the answer and how>
Answer: <concise final answer -- as few words as possible (a name, date, yes/no, etc.), \
matching the style of the question>

Graph context:
{context}

Question: {question}"""


def answer(question: str, context: str) -> dict:
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
    raw = generate(prompt, pipeline="graph_rag", stage="answer").strip()
    return parse_reasoned_answer(raw)
