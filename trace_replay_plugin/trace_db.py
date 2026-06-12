"""Request trace database for replay worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class RequestTrace:
    request_id: str
    step_token_ids: list[int]
    step_logits: list[torch.Tensor] = field(default_factory=list)
    prompt_last_token: int | None = None
    transitions: dict[int, list[int]] = field(default_factory=dict)
    transition_cursor: dict[int, int] = field(default_factory=dict)
    hidden_states: torch.Tensor | None = None
    cursor: int = 0

    def next_token_id(self) -> int | None:
        if self.cursor >= len(self.step_token_ids):
            return None
        token_id = self.step_token_ids[self.cursor]
        self.cursor += 1
        return token_id

    def reset(self) -> None:
        self.cursor = 0
        self.transition_cursor = {}


class TraceDatabase:
    """Loads request traces and serves step-wise token replay."""

    def __init__(
        self,
        traces: dict[str, RequestTrace],
        meta: dict[str, Any] | None = None,
        fallback_token_id: int = 0,
        strict_missing: bool = False,
    ) -> None:
        self.traces = traces
        self.meta = meta or {}
        self.fallback_token_id = fallback_token_id
        self.strict_missing = strict_missing

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        fallback_token_id: int = 0,
        strict_missing: bool = False,
    ) -> "TraceDatabase":
        payload = torch.load(Path(path), map_location="cpu")
        meta, records = cls._normalize_payload(payload)
        traces: dict[str, RequestTrace] = {}

        for req_id, record in records.items():
            step_token_ids = cls._extract_step_token_ids(record)
            step_logits = cls._extract_step_logits(record)
            hidden_states = record.get("hidden_states")
            prompt_last_token = cls._extract_prompt_last_token(record)
            transitions = cls._extract_transitions(record, step_token_ids)
            traces[str(req_id)] = RequestTrace(
                request_id=str(req_id),
                step_token_ids=step_token_ids,
                step_logits=step_logits,
                prompt_last_token=prompt_last_token,
                transitions=transitions,
                hidden_states=hidden_states,
            )

        return cls(
            traces=traces,
            meta=meta,
            fallback_token_id=fallback_token_id,
            strict_missing=strict_missing,
        )

    @staticmethod
    def _normalize_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        # Format A: {"meta": ..., "records": {...}}
        if isinstance(payload, dict) and "records" in payload:
            return payload.get("meta", {}), payload["records"]

        # Format B: {"req-1": logits_tensor_or_record, ...}
        if isinstance(payload, dict):
            return {}, payload

        raise ValueError("Unsupported trace db payload format.")

    @staticmethod
    def _extract_step_logits(record: Any) -> list[torch.Tensor]:
        if isinstance(record, dict):
            step_logits = record.get("step_logits")
            if isinstance(step_logits, list):
                return [x.detach().cpu().to(torch.float32) for x in step_logits]
            if isinstance(step_logits, torch.Tensor) and step_logits.dim() == 2:
                return [row.detach().cpu().to(torch.float32) for row in step_logits]
            if "next_token_logits" in record and isinstance(
                record["next_token_logits"], torch.Tensor
            ):
                return [record["next_token_logits"].detach().cpu().to(torch.float32)]
        elif isinstance(record, torch.Tensor) and record.dim() == 1:
            return [record.detach().cpu().to(torch.float32)]
        return []

    @staticmethod
    def _extract_step_token_ids(record: Any) -> list[int]:
        if isinstance(record, dict):
            if "step_token_ids" in record and isinstance(record["step_token_ids"], list):
                return [int(x) for x in record["step_token_ids"]]
            if "next_token_id" in record:
                return [int(record["next_token_id"])]

        logits = TraceDatabase._extract_step_logits(record)
        if logits:
            return [int(torch.argmax(step).item()) for step in logits]

        raise ValueError("Request trace does not provide step_token_ids or logits.")

    @staticmethod
    def _extract_prompt_last_token(record: Any) -> int | None:
        if isinstance(record, dict) and "prompt_last_token" in record:
            return int(record["prompt_last_token"])
        return None

    @staticmethod
    def _extract_transitions(
        record: Any,
        step_token_ids: list[int],
    ) -> dict[int, list[int]]:
        transitions: dict[int, list[int]] = {}
        if isinstance(record, dict) and isinstance(record.get("transitions"), list):
            for item in record["transitions"]:
                if not isinstance(item, dict):
                    continue
                if "current_token" not in item or "next_token" not in item:
                    continue
                cur = int(item["current_token"])
                nxt = int(item["next_token"])
                transitions.setdefault(cur, []).append(nxt)
            return transitions

        # Backward-compat fallback: no explicit transitions.
        # Keep empty map and rely on sequential cursor replay.
        _ = step_token_ids
        return transitions

    def get_next_token_id(self, request_id: str) -> int:
        trace = self.traces.get(request_id)
        if trace is None:
            if self.strict_missing:
                raise KeyError(f"Missing request_id in trace db: {request_id}")
            return self.fallback_token_id

        token_id = trace.next_token_id()
        if token_id is None:
            # End of trace: keep returning last token when available.
            if trace.step_token_ids:
                return trace.step_token_ids[-1]
            return self.fallback_token_id
        return token_id

    def get_next_token_by_last_token(
        self,
        request_id: str,
        current_last_token: int | None,
    ) -> int:
        trace = self.traces.get(request_id)
        if trace is None:
            if self.strict_missing:
                raise KeyError(f"Missing request_id in trace db: {request_id}")
            return self.fallback_token_id

        if (
            current_last_token is not None
            and current_last_token in trace.transitions
            and trace.transitions[current_last_token]
        ):
            next_list = trace.transitions[current_last_token]
            idx = trace.transition_cursor.get(current_last_token, 0)
            if idx >= len(next_list):
                idx = len(next_list) - 1
            token_id = next_list[idx]
            trace.transition_cursor[current_last_token] = idx + 1
            return token_id

        # Fallback to sequential replay.
        return self.get_next_token_id(request_id)

    def finish_request(self, request_id: str) -> None:
        trace = self.traces.get(request_id)
        if trace is not None:
            trace.reset()
