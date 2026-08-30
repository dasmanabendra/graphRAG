def summarize(eval_result: dict, pipeline_names: list[str]) -> str:
    records = eval_result["records"]
    n = len(records)
    lines = [f"Questions evaluated: {n}", ""]

    header = f"{'pipeline':<12} {'EM':>6} {'F1':>6} {'ret.P':>7} {'ret.R':>7} {'avg latency':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name in pipeline_names:
        em = sum(r[name]["em"] for r in records) / n
        f1 = sum(r[name]["f1"] for r in records) / n
        precision = sum(r[name]["retrieval_precision"] for r in records) / n
        recall = sum(r[name]["retrieval_recall"] for r in records) / n
        latency = sum(r[name]["latency_sec"] for r in records) / n
        lines.append(f"{name:<12} {em:>6.2f} {f1:>6.2f} {precision:>7.2f} {recall:>7.2f} {latency:>10.2f}s")

    lines.append("")
    lines.append("Call usage this run:")
    for key, count in sorted(eval_result["call_deltas"].items()):
        lines.append(f"  {key}: {count}")

    return "\n".join(lines)
