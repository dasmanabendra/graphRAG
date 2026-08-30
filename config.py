import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _gemini_key_from_credential_manager() -> str:
    """Fallback source: Windows Credential Manager (Generic Credential,
    target "gemini_api", username "gemini_api") -- same convention used by
    the amc8_qs_gen project. Only consulted when GEMINI_API_KEY isn't set
    via env/`.env`, so a plaintext override still wins if present.
    """
    try:
        import keyring

        return keyring.get_password("gemini_api", "gemini_api") or ""
    except Exception:
        return ""


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or _gemini_key_from_credential_manager()

CHAT_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

DAILY_CALL_CAP = 500

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_HOTPOTQA_PATH = DATA_DIR / "raw_hotpotqa.json"
CORPUS_PATH = DATA_DIR / "corpus.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
GRAPH_PATH = DATA_DIR / "graph.gpickle"
COMMUNITIES_PATH = DATA_DIR / "communities.json"
COMMUNITY_SUMMARIES_PATH = DATA_DIR / "community_summaries.json"
CALL_LOG_PATH = DATA_DIR / "call_log.jsonl"
RESPONSE_CACHE_DIR = DATA_DIR / "response_cache"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

DEV_N_EXAMPLES = 5
FULL_N_EXAMPLES = 25

EXTRACTION_BATCH_SIZE = 7
VECTOR_TOP_K = 5
LOUVAIN_RESOLUTION = 1.0
MIN_COMMUNITY_SIZE_FOR_SUMMARY = 3
ENTITY_FUZZY_MATCH_THRESHOLD = 0.8
MAX_LOCAL_SEARCH_NEIGHBORS = 20

DATA_DIR.mkdir(exist_ok=True)
RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
