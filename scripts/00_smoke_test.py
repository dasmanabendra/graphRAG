import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from common.embeddings import embed_text
from common.llm_client import generate


def main():
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

    print("Testing chat call...")
    text = generate("Reply with exactly the word: OK", pipeline="smoke", stage="chat")
    print(f"  chat response: {text!r}")

    print("Testing embedding call...")
    vec = embed_text("hello world", pipeline="smoke", stage="embed", task_type="RETRIEVAL_DOCUMENT")
    print(f"  embedding dim: {len(vec)}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
