#!/usr/bin/env python
"""Project collected hidden states to logits with model lm_head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input-pt", required=True, help="PT from collect_intermediate_states.py")
    p.add_argument("--output-pt", required=True, help="Output trace DB with step_logits.")
    p.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    payload = torch.load(Path(args.input_pt), map_location="cpu")
    records = payload["records"]

    dtype = to_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    model.to(args.device)

    for req_id, rec in records.items():
        hidden = rec.get("hidden_states")
        if hidden is None:
            continue
        hidden = hidden.to(device=args.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model.lm_head(hidden)[0, 0, :].detach().cpu().to(torch.float32)
        rec["step_logits"] = [logits]
        rec["step_token_ids"] = [int(torch.argmax(logits).item())]
        records[req_id] = rec

    out = {
        "meta": {
            **payload.get("meta", {}),
            "projected_by": "project_hidden_to_logits.py",
        },
        "records": records,
    }
    out_path = Path(args.output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved projected trace db to {out_path}")


if __name__ == "__main__":
    main()
