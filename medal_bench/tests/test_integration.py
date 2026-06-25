"""Integration: run the real AL loop on a tiny synthetic dataset for two budgets
(Section 14/17). Verifies same initial set across methods, cumulative budget
increments, unique selections, score logging, finite scores, and deterministic
replay from the same seed."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.runner.al_loop import run_al, RunConfig, TrainConfig


class _SynthAdapter(MedALDataset):
    name = "synthint"; modality = "synthetic"; target = "fg"
    dim = "2d"; query_unit = "image"; num_classes = 2

    def __init__(self, n=40, h=32, w=32, seed=0):
        rng = np.random.RandomState(seed)
        self._imgs = [rng.rand(1, h, w).astype(np.float32) for _ in range(n)]
        self._masks = [(rng.rand(h, w) > 0.7).astype(np.int64) for _ in range(n)]
        self._ids = [f"s{i:03d}" for i in range(n)]

    def __len__(self): return len(self._ids)
    def sample_ids(self): return list(self._ids)
    def __getitem__(self, i):
        return Sample(sample_id=self._ids[i], image=self._imgs[i], mask=self._masks[i])


def _cfg(out_jsonl):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    return RunConfig(
        policy_id="P0", policy_config={}, dataset_name="synthint", seed=1000,
        budget_plan=[3, 6], train=TrainConfig(num_iters=2, batch_size=2, image_size=32,
                                              features_per_stage=(8, 16, 32), dropout_p=0.1),
        out_jsonl=out_jsonl, device=dev, pool_cap=None, val_cap=None,
    )


def _read(path):
    return [json.loads(l) for l in open(path)]


def test_real_data_smoke_invariants(tmp_path):
    adapter = _SynthAdapter()
    out = str(tmp_path)
    recs = {}
    for pid in ["P0", "P1", "P4"]:
        cfg = _cfg(os.path.join(out, f"synthint__{pid}__s1000.jsonl"))
        cfg.policy_id = pid
        run_al(adapter, cfg)
        recs[pid] = _read(os.path.join(out, f"synthint__{pid}__s1000.jsonl"))

    # init set saved + shared (one file for all methods at this dataset+seed+n)
    init_files = os.listdir(os.path.join(out, "init_sets"))
    assert init_files == ["synthint__s1000__n3.json"], init_files
    init = json.load(open(os.path.join(out, "init_sets", init_files[0])))
    assert len(init["sample_ids"]) == 3 and len(set(init["sample_ids"])) == 3

    for pid, rr in recs.items():
        r0 = rr[0]
        assert r0["labeled_count"] == 3, (pid, r0["labeled_count"])      # 1%-analog start
        assert len(r0["selected_ids"]) == 3                              # +3 to reach budget[1]=6
        assert len(set(r0["selected_ids"])) == 3                         # unique
        assert set(r0["selected_ids"]).issubset(set(adapter.sample_ids()))
    # scoring policies must log finite scores (P0/P4 have no per-sample score -> nan by design)
    p1r0 = recs["P1"][0]
    assert p1r0["selected_scores"] and np.isfinite(np.asarray(p1r0["selected_scores"])).all()


def test_deterministic_replay_from_same_seed(tmp_path):
    adapter = _SynthAdapter()
    sel = []
    for rep in ("a", "b"):
        d = tmp_path / rep
        d.mkdir()
        cfg = _cfg(str(d / "synthint__P1__s1000.jsonl"))
        cfg.policy_id = "P1"
        run_al(adapter, cfg)
        sel.append(tuple(_read(str(d / "synthint__P1__s1000.jsonl"))[0]["selected_ids"]))
    assert sel[0] == sel[1], f"non-deterministic replay: {sel}"
