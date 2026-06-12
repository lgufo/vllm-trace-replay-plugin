import tempfile
from pathlib import Path

import torch

from trace_replay_plugin.trace_db import TraceDatabase


def test_trace_db_step_replay_and_finish_reset():
    payload = {
        "meta": {"model": "dummy"},
        "records": {
            "req-1": {"step_token_ids": [10, 11, 12]},
            "req-2": {"step_token_ids": [20]},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trace.pt"
        torch.save(payload, path)

        db = TraceDatabase.from_file(path)
        assert db.get_next_token_id("req-1") == 10
        assert db.get_next_token_id("req-1") == 11
        assert db.get_next_token_id("req-2") == 20
        # End-of-trace fallback is last token.
        assert db.get_next_token_id("req-2") == 20

        db.finish_request("req-1")
        assert db.get_next_token_id("req-1") == 10


def test_trace_db_tensor_logits_payload():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trace.pt"
        torch.save({"req-x": torch.tensor([0.1, 0.2, 1.0])}, path)
        db = TraceDatabase.from_file(path)
        assert db.get_next_token_id("req-x") == 2
