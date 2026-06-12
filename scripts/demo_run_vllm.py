#!/usr/bin/env python3
"""Minimal end-to-end demo: vLLM + trace-replay plugin + TRACE_REPLAY_DB_PATH.

Prerequisites
-------------
1) Install this package in the same environment as vLLM::

       pip install -e /path/to/vllm-trace-replay-plugin

2) Run this script *before* importing vLLM in your own code, or execute this
   file directly so env vars are set first.

Environment
-----------
- ``VLLM_PLUGINS=trace_replay`` — only this OOT platform plugin is loaded when
  multiple platform plugins are installed (e.g. fake-logits + trace-replay).
- ``TRACE_REPLAY_DB_PATH`` — path to ``trace_db.pt`` (see ``trace_db.py``).
- ``TRACE_REPLAY_REPORT_PATH`` — optional JSONL schedule log (demo sets a temp file).

Request IDs
-----------
``LLM.generate()`` assigns string request ids ``"0"``, ``"1"``, ``"2"``, ...
from an internal counter. The default trace written by this demo uses those
keys so replay lines up without patching vLLM.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch


def write_default_trace_db(path: Path) -> None:
    """Synthetic trace for the first three LLM.generate() request ids."""
    long_tail = list(range(1000, 1100))
    payload = {
        "meta": {
            "demo": True,
            "description": "Synthetic replay trace for demo_run_vllm.py",
        },
        "records": {
            "0": {"step_token_ids": long_tail},
            "1": {"step_token_ids": [x + 10 for x in long_tail]},
            "2": {"step_token_ids": [x + 20 for x in long_tail]},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=os.environ.get("TRACE_REPLAY_DB_PATH", ""),
        help="Path to trace_db.pt; if missing, a default file is written under /tmp.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HF model id (weights loaded with load_format=dummy).",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None
    if db_path is None or not db_path.is_file():
        tmp = Path(tempfile.mkdtemp(prefix="vllm-trace-replay-demo-"))
        db_path = tmp / "trace_db.pt"
        write_default_trace_db(db_path)
        print(f"[demo] wrote default trace db to {db_path}", file=sys.stderr)

    report_dir = Path(tempfile.mkdtemp(prefix="vllm-trace-replay-report-"))
    report_path = report_dir / "schedule_report.jsonl"

    # Must be set before vLLM resolves current_platform / loads plugins.
    os.environ["VLLM_PLUGINS"] = "trace_replay"
    os.environ["TRACE_REPLAY_DB_PATH"] = str(db_path)
    os.environ["TRACE_REPLAY_REPORT_PATH"] = str(report_path)

    # Import vLLM after env is configured.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompts = [
        "Hello trace replay.",
        "Second request.",
        "Third request.",
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    print("[demo] generation results (request_id matches trace keys 0,1,2):")
    for out in outputs:
        o0 = out.outputs[0] if out.outputs else None
        token_ids = list(o0.token_ids) if o0 else []
        text_preview = (o0.text if o0 else "")[:120]
        print(f"  request_id={out.request_id!r} n_tokens={len(token_ids)} preview={text_preview!r}")

    print(f"[demo] schedule report: {report_path}")
    if report_path.is_file():
        lines = report_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"[demo] report lines: {len(lines)} (showing first 3)")
        for line in lines[:3]:
            print(f"  {line[:500]}{'...' if len(line) > 500 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
