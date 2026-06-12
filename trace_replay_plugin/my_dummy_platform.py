"""Out-of-tree platform for trace replay worker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig


class MyDummyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "TraceReplayDevice"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "CPU"
    device_control_env_var: str = "VLLM_TRACE_REPLAY_VISIBLE_DEVICES"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # The critical hook: bind worker implementation early so executor uses it.
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = (
                "trace_replay_plugin.my_dummy_worker.MyDummyWorker"
            )

        # Keep runtime conservative and deterministic for replay mode.
        vllm_config.scheduler_config.async_scheduling = False
        # Replay worker has no real KV cache. Disable cache-dependent scheduler
        # features to avoid coordinator initialization paths that expect real
        # attention-group KV specs.
        vllm_config.cache_config.enable_prefix_caching = False
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
        vllm_config.compilation_config.custom_ops = []

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        # Replay worker does not execute real attention, but vLLM still expects
        # a resolvable attention backend class in platform flow.
        return "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.base_device_communicator."
            "DeviceCommunicatorBase"
        )

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"{cls.device_name}-{device_id}"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"trace-replay-{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return 0

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # CPU replay mode has no mutable device context.
        return

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return 0.0

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_base.PunicaWrapperBase"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls, backend, prefix_store, group_rank, group_size, timeout
    ):
        raise NotImplementedError(
            "Trace replay platform does not initialize custom torch dist groups."
        )

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype not in cls().supported_dtypes:
            raise ValueError(f"dtype {dtype} is not supported on trace replay platform.")
