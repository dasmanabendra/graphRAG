import json

from datasets import load_dataset

import config


def fetch_examples(n: int) -> list[dict]:
    """Stream N examples from the HotpotQA distractor split and cache them
    to disk. Streaming + .take(n) avoids downloading the full validation
    split (7,405 rows) just to use a handful.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    examples = list(ds.take(n))
    config.RAW_HOTPOTQA_PATH.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Fetched {len(examples)} HotpotQA examples -> {config.RAW_HOTPOTQA_PATH}")
    return examples


def load_cached_examples() -> list[dict]:
    return json.loads(config.RAW_HOTPOTQA_PATH.read_text(encoding="utf-8"))
