"""frozen_v3 orchestration: done-by-round-count (lease-race safe) and the per-cell
timeout that kills a hung child instead of wedging the worker."""
from __future__ import annotations

import json
import os
import sys
import time

from medal_bench.runner.dispatch import _cell_done, _run_cell, CELL_TIMEOUT_SEC


def _write(path, n_records, total_rounds):
    with open(path, "w") as f:
        for r in range(n_records):
            f.write(json.dumps({"round": r, "total_rounds": total_rounds}) + "\n")


def test_cell_done_by_round_count(tmp_path):
    complete = str(tmp_path / "c.jsonl"); _write(complete, 6, 6)
    short = str(tmp_path / "s.jsonl"); _write(short, 3, 6)        # truncated
    assert _cell_done(complete) is True
    assert _cell_done(short) is False                            # NOT done -> re-runnable
    assert _cell_done(str(tmp_path / "missing.jsonl")) is False


def test_run_cell_timeout_kills_and_returns_sentinel():
    # child sleeps far longer than the (tiny) timeout -> must be killed, returns -9 fast.
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    t0 = time.time()
    rc = _run_cell(cmd, dict(os.environ), timeout=2)
    dt = time.time() - t0
    assert rc == -9
    assert dt < 15           # killed promptly, did not wait the full 30s


def test_run_cell_normal_returns_rc():
    rc = _run_cell([sys.executable, "-c", "raise SystemExit(0)"], dict(os.environ),
                   timeout=CELL_TIMEOUT_SEC)
    assert rc == 0
