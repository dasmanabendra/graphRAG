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


def _cache_key(prompt: str, model: str, schema_repr: str) -> str:
    raw = json.dumps({"prompt": prompt, "model": model, "schema": schema_repr}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return config.RESPONSE_CACHE_DIR / f"{key}.txt"


def generate(
    prompt: str,
    *,
    pipeline: str,
    stage: str,
    response_schema=None,
    model: str = config.CHAT_MODEL,
) -> str:
    """Single chat call to Gemini. Response is cached on disk keyed by
    (prompt, model, schema) so re-running the same prompt during debugging
    costs nothing after the first call. Every non-cached call is logged
    through the shared CallTracker and checked against the daily cap first.
    """
    schema_repr = repr(response_schema)
    key = _cache_key(prompt, model, schema_repr)
    cache_file = _cache_path(key)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    tracker.check_budget("chat")

    gen_config = None
    if response_schema is not None:
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    client = _get_client()
    response = call_with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt, config=gen_config)
    )
    text = response.text or ""

    tracker.log_call("chat", pipeline, stage, meta={"model": model, "prompt_chars": len(prompt)})
    cache_file.write_text(text, encoding="utf-8")
    return text
