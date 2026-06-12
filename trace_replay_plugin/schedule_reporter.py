"""Step-level schedule reporting utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ScheduleReporter:
    def __init__(self, jsonl_path: str | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.step_idx = 0
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def report(
        self,
        scheduled_req_ids: list[str],
        new_req_ids: list[str],
        cached_req_ids: list[str],
        finished_req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
    ) -> None:
        if not self.enabled:
            return

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_idx": self.step_idx,
            "scheduled_req_ids": scheduled_req_ids,
            "new_req_ids": new_req_ids,
            "cached_req_ids": cached_req_ids,
            "finished_req_ids": finished_req_ids,
            "num_scheduled_tokens": num_scheduled_tokens,
        }

        logger.info(
            "trace-replay step=%d scheduled=%s new=%s cached=%s finished=%s",
            self.step_idx,
            scheduled_req_ids,
            new_req_ids,
            cached_req_ids,
            finished_req_ids,
        )

        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        self.step_idx += 1
