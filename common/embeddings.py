import hashlib
import json
from typing import Optional

from google import genai
from google.genai import types

import config
from common.call_budget import tracker
from common.retry import call_with_retry

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _cache_key(text: str, model: str, task_type: str, dim: int) -> str:
    raw = json.dumps({"text": text, "model": model, "task_type": task_type, "dim": dim}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return config.RESPONSE_CACHE_DIR / f"emb_{key}.json"


def embed_texts(
    texts: list[str],
    *,
    pipeline: str,
    stage: str,
    task_type: str,
    model: str = config.EMBEDDING_MODEL,
    dim: int = config.EMBEDDING_DIM,
    batch_size: int = 16,
) -> list[list[float]]:
    """Embed a list of texts, batching uncached texts into as few calls as
    possible. Each text is cached individually keyed on (text, model,
    task_type, dim) so corpus overlap across runs never re-costs a call.
    """
    results: list[Optional[list[float]]] = [None] * len(texts)
    pending_idx: list[int] = []
    pending_texts: list[str] = []

    for i, text in enumerate(texts):
        cache_file = _cache_path(_cache_key(text, model, task_type, dim))
        if cache_file.exists():
            results[i] = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            pending_idx.append(i)
            pending_texts.append(text)

    client = _get_client()
    embed_config = types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type)

    for start in range(0, len(pending_texts), batch_size):
        batch_idx = pending_idx[start : start + batch_size]
        batch_texts = pending_texts[start : start + batch_size]

        tracker.check_budget("embedding")
        response = call_with_retry(
            lambda bt=batch_texts: client.models.embed_content(model=model, contents=bt, config=embed_config)
        )
        tracker.log_call("embedding", pipeline, stage, meta={"model": model, "batch_size": len(batch_texts)})

        for idx, embedding in zip(batch_idx, response.embeddings):
            values = list(embedding.values)
            results[idx] = values
            _cache_path(_cache_key(texts[idx], model, task_type, dim)).write_text(
                json.dumps(values), encoding="utf-8"
            )

    return results  # type: ignore[return-value]


def embed_text(text: str, **kwargs) -> list[float]:
    return embed_texts([text], **kwargs)[0]
