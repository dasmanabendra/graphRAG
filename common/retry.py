import re
import time
from typing import Callable, TypeVar

from google.genai import errors

T = TypeVar("T")


def _extract_retry_delay(error: errors.APIError, default: float) -> float:
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", details)
        for item in inner.get("details", []) or []:
            if str(item.get("@type", "")).endswith("RetryInfo"):
                match = re.match(r"([\d.]+)s", str(item.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    return default


def call_with_retry(fn: Callable[[], T], *, max_attempts: int = 6, base_delay: float = 5.0) -> T:
    """Retries on HTTP 429 (free-tier rate limit) with backoff, using the
    server-suggested retryDelay when present. Any other error is raised
    immediately -- only rate limiting is worth retrying automatically.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429 or attempt == max_attempts - 1:
                raise
            delay = _extract_retry_delay(e, default=base_delay * (2**attempt)) + 1.0
            print(f"[retry] rate limited (attempt {attempt + 1}/{max_attempts}), waiting {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises
