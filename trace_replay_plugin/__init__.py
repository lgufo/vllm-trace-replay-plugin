"""Trace replay platform plugin entrypoint."""

from __future__ import annotations


def trace_replay_platform_plugin() -> str | None:
    """Return the platform class path for vLLM plugin loading."""
    return "trace_replay_plugin.my_dummy_platform.MyDummyPlatform"
