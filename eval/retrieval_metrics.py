def retrieval_precision_recall(retrieved_titles: list[str], gold_titles: list[str]) -> tuple[float, float]:
    retrieved_set = set(retrieved_titles)
    gold_set = set(gold_titles)

    precision = len(retrieved_set & gold_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(retrieved_set & gold_set) / len(gold_set) if gold_set else 0.0
    return precision, recall
