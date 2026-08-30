import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

import config


class CallTracker:
    def __init__(self, log_path: Path = config.CALL_LOG_PATH, daily_cap: int = config.DAILY_CALL_CAP):
        self.log_path = log_path
        self.daily_cap = daily_cap

    def _today_entries(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        today = date.today().isoformat()
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("date") == today:
                    entries.append(entry)
        return entries

    def today_count(self, bucket: str) -> int:
        return sum(1 for e in self._today_entries() if e.get("bucket") == bucket)

    def check_budget(self, bucket: str = "chat") -> None:
        count = self.today_count(bucket)
        if count >= self.daily_cap:
            raise RuntimeError(
                f"Daily call cap reached for bucket '{bucket}': {count}/{self.daily_cap}. "
                "Refusing further calls today."
            )
        if count >= self.daily_cap * 0.9:
            print(f"[call_budget] WARNING: {count}/{self.daily_cap} '{bucket}' calls used today.")

    def log_call(self, bucket: str, pipeline: str, stage: str, meta: Optional[dict] = None) -> None:
        entry = {
            "date": date.today().isoformat(),
            "timestamp": time.time(),
            "bucket": bucket,
            "pipeline": pipeline,
            "stage": stage,
            "meta": meta or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self._today_entries():
            key = f"{entry.get('pipeline', '?')}/{entry.get('bucket', '?')}"
            totals[key] = totals.get(key, 0) + 1
        return totals

    def stage_counts(self) -> dict[str, int]:
        """Like totals(), but keyed by pipeline/stage instead of
        pipeline/bucket -- lets the UI distinguish e.g. graph_rag's
        extraction calls from its summarization calls.
        """
        counts: dict[str, int] = {}
        for entry in self._today_entries():
            key = f"{entry.get('pipeline', '?')}/{entry.get('stage', '?')}"
            counts[key] = counts.get(key, 0) + 1
        return counts


tracker = CallTracker()
