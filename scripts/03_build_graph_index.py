import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.call_budget import tracker
from graph_rag.pipeline import run_index

if __name__ == "__main__":
    run_index()
    print("Call totals today:", tracker.totals())
