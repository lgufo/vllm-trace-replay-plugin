#!/usr/bin/env python
"""Collect request intermediate states and optional logits for trace replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-pt", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--num-decode-steps",
        type=int,
        default=1,
        help="How many autoregressive decode steps to sample for each request.",
    )
    return p.parse_args()


def to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "request_id" not in row or "prompt" not in row:
                raise ValueError(f"line {line_no} missing request_id/prompt")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    dtype = to_dtype(args.dtype)
    requests = load_requests(Path(args.input_jsonl))
    if not requests:
        raise ValueError("No requests loaded from input JSONL.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    records: dict[str, dict[str, Any]] = {}
    for row in requests:
        req_id = str(row["request_id"])
        prompt = str(row["prompt"])
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        step_token_ids: list[int] = []
        step_logits: list[torch.Tensor] = []
        transitions: list[dict[str, int]] = []
        hidden_states: torch.Tensor | None = None

        with torch.no_grad():
            for _ in range(args.num_decode_steps):
                current_last_token = int(input_ids[0, -1].item())
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                last_hidden = out.hidden_states[-1][0, -1, :].detach().cpu().to(torch.float32)
                logits = out.logits[0, -1, :].detach().cpu().to(torch.float32)
                token_id = int(torch.argmax(logits).item())

                if hidden_states is None:
                    hidden_states = last_hidden
                step_logits.append(logits)
                step_token_ids.append(token_id)
                transitions.append(
                    {
                        "current_token": current_last_token,
                        "next_token": token_id,
                    }
                )

                next_token = torch.tensor([[token_id]], device=input_ids.device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    extra = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, extra], dim=1)

        records[req_id] = {
            "prompt": prompt,
            "prompt_last_token": int(enc["input_ids"][0, -1].item()),
            "hidden_states": hidden_states,
            "step_logits": step_logits,
            "step_token_ids": step_token_ids,
            "transitions": transitions,
        }

    payload = {
        "meta": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "num_decode_steps": args.num_decode_steps,
            "num_requests": len(records),
        },
        "records": records,
    }

    out_path = Path(args.output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved trace db to {out_path}")


if __name__ == "__main__":
    main()
