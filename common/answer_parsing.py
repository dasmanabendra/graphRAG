import re

_ANSWER_PATTERN = re.compile(r"(?is)\*{0,2}answer\*{0,2}\s*:\s*(.*)")
_REASONING_PREFIX = re.compile(r"(?i)^\*{0,2}reasoning\*{0,2}\s*:\s*")


def parse_reasoned_answer(raw: str) -> dict:
    """Split a model response formatted as 'Reasoning: ...\\nAnswer: ...' into parts.

    Falls back to treating the whole response as the answer (empty reasoning) if
    the model didn't follow the format, so downstream EM/F1 scoring -- which reads
    only the answer -- never breaks on a malformed response.
    """
    raw = raw.strip()
    matches = list(_ANSWER_PATTERN.finditer(raw))
    if not matches:
        return {"answer": raw, "reasoning": "", "raw": raw}

    last = matches[-1]
    answer = last.group(1).strip()
    reasoning = _REASONING_PREFIX.sub("", raw[: last.start()].strip()).strip()
    return {"answer": answer, "reasoning": reasoning, "raw": raw}
