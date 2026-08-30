import time
from typing import Callable, Optional

from common.call_budget import tracker
from data_prep.load_hotpotqa import load_cached_examples
from eval.retrieval_metrics import retrieval_precision_recall
from eval.scoring import exact_match_score, f1_score


def gold_titles_for(example: dict) -> list[str]:
    return list(dict.fromkeys(example["supporting_facts"]["title"]))


def run_eval(
    pipelines: dict[str, Callable[[str], dict]],
    examples: Optional[list[dict]] = None,
) -> dict:
    """Run each pipeline over the same question set.

    pipelines: {name: answer_question(question) -> {"prediction": str, "chunk_ids": list[str]}}
    Returns {"records": [...], "call_deltas": {pipeline_name: {bucket/stage: count}}}
    """
    if examples is None:
        examples = load_cached_examples()

    totals_before = tracker.totals()

    records = []
    for example in examples:
        question = example["question"]
        gold_answer = example["answer"]
        gold_titles = gold_titles_for(example)

        record = {
            "id": example["id"],
            "question": question,
            "gold_answer": gold_answer,
            "gold_titles": gold_titles,
        }

        for name, pipeline_fn in pipelines.items():
            start = time.time()
            result = pipeline_fn(question)
            latency = time.time() - start

            precision, recall = retrieval_precision_recall(result["chunk_ids"], gold_titles)

            record[name] = {
                "prediction": result["prediction"],
                "chunk_ids": result["chunk_ids"],
                "em": exact_match_score(result["prediction"], gold_answer),
                "f1": f1_score(result["prediction"], gold_answer),
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "latency_sec": latency,
            }

        records.append(record)

    totals_after = tracker.totals()
    call_deltas: dict[str, int] = {}
    for key in set(totals_before) | set(totals_after):
        delta = totals_after.get(key, 0) - totals_before.get(key, 0)
        if delta:
            call_deltas[key] = delta

    return {"records": records, "call_deltas": call_deltas}
