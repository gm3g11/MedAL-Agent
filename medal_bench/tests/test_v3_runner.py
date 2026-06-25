"""frozen_v3 runner tests: component seeding + seed logging, schema v3, budget
denominator provenance, and the always-on prediction-saving round-trip."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.runner.al_loop import run_al, RunConfig, TrainConfig
from medal_bench.runner.seeds import component_seeds


class _Synth(MedALDataset):
    name = "synthv3"; modality = "synthetic"; target = "fg"
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


def _run(out_dir, **cfg_kw):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(
        policy_id="P1", policy_config={}, dataset_name="synthv3", seed=1000,
        budget_plan=[3, 6], train=TrainConfig(num_iters=2, batch_size=2, image_size=32,
                                              features_per_stage=(8, 16, 32), dropout_p=0.1),
        out_jsonl=os.path.join(out_dir, "synthv3__P1__s1000.jsonl"), device=dev,
        pool_cap=None, val_cap=None, **cfg_kw,
    )
    run_al(_Synth(), cfg)
    return [json.loads(l) for l in open(cfg.out_jsonl)], cfg


def test_component_seeds_deterministic_and_distinct():
    a, b, c = component_seeds(1234), component_seeds(1234), component_seeds(1235)
    assert a == b and a != c
    assert set(a) == {"model_init_seed", "loader_seed", "query_seed", "dropout_seed"}
    assert len(set(a.values())) == 4          # four well-separated streams


def test_schema_v3_seeds_and_budget_denominator_logged(tmp_path):
    recs, cfg = _run(str(tmp_path))
    for r, rec in enumerate(recs):
        assert rec["schema_version"] == "v3"
        assert rec["round_seed"] == cfg.seed + r
        cs = rec["component_seeds"]
        assert cs == component_seeds(cfg.seed + r)          # logged == derived
        assert rec["total_rounds"] == len(cfg.budget_plan)
        bd = rec["budget_denominator"]
        for key in ("full_train_N", "actual_AL_pool_N", "fraction_of_AL_pool",
                    "fraction_of_full_train", "budget_plan"):
            assert key in bd
        assert bd["fraction_of_AL_pool"] == cfg.budget_plan[-1] / max(1, bd["actual_AL_pool_N"])


def test_predictions_saved_and_roundtrip(tmp_path):
    recs, cfg = _run(str(tmp_path), save_predictions=True, save_logits=True)
    final = len(recs) - 1
    for r, rec in enumerate(recs):
        p = rec["predictions_path"]
        # frozen_v5: predictions are saved ONLY on the final round (NFS-write reduction);
        # the offline surface backfill consumes only the final-round npz.
        if r != final:
            assert p == "", f"round {r} (non-final) should not save predictions in v5"
            continue
        assert p and os.path.exists(p), "final-round predictions missing"
        z = np.load(p, allow_pickle=True)
        n = len(z["sample_ids"])
        assert z["pred"].shape == (n, 32, 32) and z["pred"].dtype == np.uint8
        assert z["gt"].shape == (n, 32, 32)
        assert z["valid_bbox"].shape == (n, 4)
        assert z["probs"].shape == (n, 2, 32, 32) and z["probs"].dtype == np.float16
        assert int(z["pred"].max()) < _Synth().num_classes   # pred values are class ids < C
        assert z["spacing"].item() == "unavailable"          # spacing slot for mm backfill
        # heavy preds must NOT be in the JSONL metrics dict
        assert "_preds" not in rec["metrics"]


def test_predictions_off_by_default(tmp_path):
    recs, _ = _run(str(tmp_path))
    assert all(rec["predictions_path"] == "" for rec in recs)
    assert not os.path.isdir(os.path.join(str(tmp_path), "predictions"))
