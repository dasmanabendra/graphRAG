import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data_prep.build_corpus import build_corpus
from data_prep.load_hotpotqa import fetch_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.DEV_N_EXAMPLES)
    args = parser.parse_args()

    examples = fetch_examples(args.n)
    build_corpus(examples)


if __name__ == "__main__":
    main()
