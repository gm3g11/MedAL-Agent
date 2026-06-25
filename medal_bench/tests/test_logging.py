"""Logging-upgrade tests (task 1): after a real AL run the trajectory records
must carry candidate-score sidecars, checkpoint hash, method version/ablation
flag, feature-cache keys, SAM provenance, initial labeled set, and budget
bookkeeping."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.runner.al_loop import run_al, RunConfig, TrainConfig


class _Synth(MedALDataset):
    name = "synthlog"; modality = "synthetic"; target = "fg"
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


def _run(pid, out_dir):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(
        policy_id=pid, policy_config={}, dataset_name="synthlog", seed=1000,
        budget_plan=[3, 6], train=TrainConfig(num_iters=2, batch_size=2, image_size=32,
                                              features_per_stage=(8, 16, 32), dropout_p=0.1),
        out_jsonl=os.path.join(out_dir, f"synthlog__{pid}__s1000.jsonl"), device=dev,
        pool_cap=None, val_cap=None, save_checkpoints=True,
    )
    run_al(_Synth(), cfg)
    recs = [json.loads(l) for l in open(cfg.out_jsonl)]
    return recs, out_dir


def test_logging_complete_after_run(tmp_path):
    out = str(tmp_path)
    recs, _ = _run("P1", out)
    r0 = recs[0]

    # provenance present on every round
    for rec in recs:
        assert rec["schema_version"] == "v3"
        assert rec["ckpt_hash"] and len(rec["ckpt_hash"]) == 64
        assert rec["config_hash"] and len(rec["config_hash"]) == 64
        assert rec["method_version"] == "1.0"
        assert rec["is_ablation"] is False
        assert rec["cumulative_budget"] in (3, 6)
    # round-0 query bookkeeping
    assert r0["cumulative_budget"] == 3 and r0["incremental_query_count"] == 3
    assert r0["initial_labeled_ids"] and len(r0["initial_labeled_ids"]) == 3
    assert os.path.exists(r0["initial_labeled_path"])
    assert recs[1]["initial_labeled_ids"] == []  # only round 0 carries them

    # checkpoint saved to disk + hash matches the saved file's content hash is plausible
    assert r0["ckpt_path"] and os.path.exists(r0["ckpt_path"])

    # candidate-score sidecar exists and is well-formed
    p = r0["candidate_scores_path"]
    assert p and os.path.exists(p)
    cs = json.load(open(p))
    assert len(cs["candidate_ids"]) == r0["candidate_count"]
    assert len(cs["candidate_scores"]) == r0["candidate_count"]
    # P1 is a scoring method -> finite candidate scores
    assert np.isfinite(np.asarray(cs["candidate_scores"])).all()


def test_logging_feature_cache_keys_and_ablation_flag(tmp_path):
    # P3 -> task_unet key; P4b -> badge_grad key + ablation flag; P7 -> foundation key
    r3 = _run("P3", str(tmp_path / "p3"))[0][0]
    assert "task_unet" in r3["feature_cache_keys"]

    r4b = _run("P4b", str(tmp_path / "p4b"))[0][0]
    assert r4b["is_ablation"] is True
    assert "badge_grad" in r4b["feature_cache_keys"]

    r7 = _run("P7", str(tmp_path / "p7"))[0][0]
    assert "foundation" in r7["feature_cache_keys"]
    # SAM provenance fields are present (empty under the stub extractor; populated under --foundation sam)
    assert "sam_model_type" in r7 and "sam_checkpoint" in r7
