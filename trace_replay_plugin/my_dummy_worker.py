"""Trace replay worker implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from trace_replay_plugin.schedule_reporter import ScheduleReporter
from trace_replay_plugin.trace_db import TraceDatabase

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
else:
    GrammarOutput = object
    SchedulerOutput = object
    KVCacheConfig = object
    KVCacheSpec = object

logger = init_logger(__name__)


class _NoOpModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Trace replay worker does not run real forward passes.")


class MyDummyWorker(WorkerBase):
    """Worker that replays per-request traces without hardware inference."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.device = torch.device("cpu")
        self.model_runner = _NoOpModel()

        trace_path = os.environ.get("TRACE_REPLAY_DB_PATH")
        if not trace_path:
            raise ValueError("TRACE_REPLAY_DB_PATH must be set for trace replay worker.")

        strict_missing = os.environ.get("TRACE_REPLAY_STRICT_MISSING", "0") == "1"
        fallback_token_id = int(os.environ.get("TRACE_REPLAY_FALLBACK_TOKEN_ID", "0"))
        report_path = os.environ.get(
            "TRACE_REPLAY_REPORT_PATH", "trace_replay_schedule_report.jsonl"
        )

        self.trace_db = TraceDatabase.from_file(
            path=trace_path,
            strict_missing=strict_missing,
            fallback_token_id=fallback_token_id,
        )
        self.req_last_token: dict[str, int | None] = {}
        self.reporter = ScheduleReporter(jsonl_path=report_path, enabled=True)
        self.kv_cache_config: KVCacheConfig | None = None

        logger.info(
            "Trace replay worker initialized: trace_path=%s, strict_missing=%s, fallback=%d",
            trace_path,
            strict_missing,
            fallback_token_id,
        )

    def init_device(self) -> None:
        logger.info("MyDummyWorker(rank=%d) initialized on %s", self.rank, self.device)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Compatibility no-op for APIs that still call initialize_cache.
        logger.info(
            "initialize_cache called in replay mode (gpu_blocks=%d, cpu_blocks=%d)",
            num_gpu_blocks,
            num_cpu_blocks,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        logger.info("MyDummyWorker load_model no-op (replay mode).")

    def get_model(self) -> nn.Module:
        return self.model_runner

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        # Replay mode does not allocate real KV cache.
        return {}

    def determine_available_memory(self) -> int:
        # No hardware memory profiling needed in replay mode.
        return 0

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        self.kv_cache_config = kv_cache_config
        logger.info("MyDummyWorker received kv_cache_config in replay mode.")

    def compile_or_warm_up_model(self) -> float:
        return 0.0

    def get_cache_block_size_bytes(self) -> int:
        return 0

    def get_supported_tasks(self) -> tuple[str, ...]:
        # Trace replay worker only supports text generation style outputs.
        return ("generate",)

    def _extract_req_ids(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[list[str], list[str], list[str]]:
        new_req_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)
        scheduled_keys = list(scheduler_output.num_scheduled_tokens.keys())

        # Use scheduler token-map order as primary execution order.
        ordered: list[str] = []
        for req_id in scheduled_keys + new_req_ids + cached_req_ids:
            if req_id not in ordered:
                ordered.append(req_id)
        return ordered, new_req_ids, cached_req_ids

    def _cleanup_finished(self, scheduler_output: "SchedulerOutput") -> list[str]:
        finished_req_ids = list(scheduler_output.finished_req_ids)
        for req_id in finished_req_ids:
            self.trace_db.finish_request(self._resolve_trace_req_id(req_id))
            self.req_last_token.pop(req_id, None)
        return finished_req_ids

    def _resolve_trace_req_id(self, req_id: str) -> str:
        """Map scheduler/runtime req_id to trace-db req_id.

        vLLM may append runtime suffixes to external request ids
        (e.g. "req-001-93fdb26d"). Trace DB usually stores the external id
        ("req-001"), so we attempt a conservative suffix strip fallback.
        """
        if req_id in self.trace_db.traces:
            return req_id

        if "-" in req_id:
            base_req_id = req_id.rsplit("-", 1)[0]
            if base_req_id in self.trace_db.traces:
                return base_req_id

        return req_id

    def _extract_new_req_last_tokens(
        self, scheduler_output: "SchedulerOutput"
    ) -> dict[str, int | None]:
        last_tokens: dict[str, int | None] = {}
        for req in scheduler_output.scheduled_new_reqs:
            token_ids = req.prompt_token_ids
            if token_ids:
                last_tokens[req.req_id] = int(token_ids[-1])
            else:
                last_tokens[req.req_id] = None
        return last_tokens

    def execute_model(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput | None:
        finished_req_ids = self._cleanup_finished(scheduler_output)
        req_ids, new_req_ids, cached_req_ids = self._extract_req_ids(scheduler_output)
        new_req_last_tokens = self._extract_new_req_last_tokens(scheduler_output)

        if not req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={})

        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            trace_req_id = self._resolve_trace_req_id(req_id)
            current_last_token = self.req_last_token.get(req_id)
            if current_last_token is None:
                current_last_token = new_req_last_tokens.get(req_id)
            if current_last_token is None:
                trace = self.trace_db.traces.get(trace_req_id)
                if trace is not None:
                    current_last_token = trace.prompt_last_token

            token_id = self.trace_db.get_next_token_by_last_token(
                trace_req_id, current_last_token
            )
            sampled_token_ids.append([int(token_id)])
            self.req_last_token[req_id] = int(token_id)

        self.reporter.report(
            scheduled_req_ids=req_ids,
            new_req_ids=new_req_ids,
            cached_req_ids=cached_req_ids,
            finished_req_ids=finished_req_ids,
            num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
        )

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
        )

    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput:
        # execute_model already returns sampled_token_ids.
        return ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def add_lora(self, lora_request: Any) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()
