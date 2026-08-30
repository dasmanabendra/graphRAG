import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from eval.report import summarize
from eval.run_eval import run_eval
from graph_rag.pipeline import answer_question as graph_answer_question
from hybrid_rag.pipeline import answer_question as hybrid_answer_question
from vector_rag.pipeline import answer_question as vector_answer_question


def main():
    result = run_eval(
        {
            "vector_rag": vector_answer_question,
            "graph_rag": graph_answer_question,
            "hybrid_rag": hybrid_answer_question,
        }
    )

    config.EVAL_RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = summarize(result, ["vector_rag", "graph_rag", "hybrid_rag"])
    print(report)
    print(f"\nFull results saved to {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
