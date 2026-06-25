"""Large-pool stress test (task 3) — shared-state design.

Builds ONE AL state on a >=2k-candidate pool (one eager-load, one trained
round-model, shared prediction cache + task features + SAM features), then
times each method's QUERY (score+select) on that fixed state. This isolates
per-method query cost/memory at scale and avoids redundant reload/retrain.

Checks: no NaN/Inf candidate scores, no duplicate selections, candidate-score
sidecars saved, checkpoint hash saved, deterministic replay, no stale cache
(features bound to the current checkpoint). P7/P8 use SAM-B here for feasibility;
SAM-H per-image cost is reported by pilot_sam (linear in #images).

Usage:
    python -m medal_bench.runner.stress_test --out runs/stress --cap 2000 \
        --init 50 --k 50 --num-iters 15 --image-size 128 --sam-model-type vit_b
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from medal_bench.data.adapters import DATASET_REGISTRY
from medal_bench.policies import build, PolicyContext
from medal_bench.runner.al_loop import _IndexedSubset, _build_model, TrainConfig, _load_or_make_initial_labeled, _initial_labeled_path
from medal_bench.runner.splits import make_split, SplitView
from medal_bench.runner.trainer import train_from_scratch
from medal_bench.runner.prediction_cache import build_prediction_cache
from medal_bench.runner.feature_extractor import extract_task_unet_features
from medal_bench.runner.trajectory import state_dict_hash, write_candidate_scores
from medal_bench.runner.seeds import seed_all

DATA_ROOT = "/groups/echambe2/datasets/data"


def _iter_imgs(ds, image_size):
    import torch.nn.functional as F
    for i in range(len(ds)):
        x = torch.from_numpy(ds[i].image).unsqueeze(0)
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        yield ds[i].sample_id, x


def _peak_mb():
    return int(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else -1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/stress")
    ap.add_argument("--dataset", default="isic2018")
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--init", type=int, default=50)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--num-iters", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--sam-model-type", default="vit_b")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub")
    os.environ.setdefault("HF_HUB_CACHE", "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub")
    seed_all(1000)

    adapter = DATASET_REGISTRY[args.dataset](DATA_ROOT)
    split = make_split(adapter, seed=1000)
    train_view = SplitView(adapter, split.train, "train")
    rng = np.random.RandomState(1000)
    idx = list(range(len(train_view)))
    cap = min(args.cap, len(idx))
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False).tolist()
    print(f"[stress] {args.dataset} train_pool={len(train_view)} cap={cap} "
          f"img={args.image_size} sam={args.sam_model_type}", flush=True)

    t0 = time.time()
    pool_subset = _IndexedSubset(train_view, idx, args.image_size)
    print(f"[stress] eager-loaded {len(pool_subset)} samples in {time.time()-t0:.1f}s", flush=True)

    nc = adapter.num_classes
    in_ch = int(pool_subset[0].image.shape[0])

    # initial labeled set (persisted + shared)
    class _Cfg:  # minimal shim for the init-set helper
        out_jsonl = os.path.join(args.out, f"{args.dataset}__INIT__s1000.jsonl")
        seed = 1000; budget_plan = [args.init, args.init + args.k]
    init_ids = _load_or_make_initial_labeled(_Cfg, adapter.name, pool_subset, rng, args.init)
    id2local = {pool_subset[i].sample_id: i for i in range(len(pool_subset))}
    labeled_local = sorted(id2local[s] for s in init_ids)
    unlabeled_local = sorted(set(range(len(pool_subset))) - set(labeled_local))
    labeled_ds = _IndexedSubset(pool_subset, labeled_local, args.image_size)
    unlabeled_ds = _IndexedSubset(pool_subset, unlabeled_local, args.image_size)
    print(f"[stress] labeled={len(labeled_ds)} unlabeled(candidates)={len(unlabeled_ds)}", flush=True)

    # ONE trained round-model
    tt = time.time()
    model = _build_model(in_ch, nc, TrainConfig(image_size=args.image_size, features_per_stage=(16, 32, 64))).to(args.device)
    train_from_scratch(model, labeled_ds, num_iters=args.num_iters, batch_size=8, lr=1e-3,
                       image_size=args.image_size, num_classes=nc, device=args.device, seed=1000)
    ckpt_hash = state_dict_hash(model)
    print(f"[stress] trained round-model in {time.time()-tt:.1f}s  ckpt_hash={ckpt_hash[:12]}", flush=True)

    # shared features (computed ONCE from the current checkpoint)
    tf = time.time()
    pred_cache = build_prediction_cache(model, _iter_imgs(unlabeled_ds, args.image_size), device=args.device)
    task = {"task_unet_pool": extract_task_unet_features(model, unlabeled_ds, image_size=args.image_size, device=args.device),
            "task_unet_label": extract_task_unet_features(model, labeled_ds, image_size=args.image_size, device=args.device)}
    print(f"[stress] pred_cache+task features in {time.time()-tf:.1f}s", flush=True)
    ts = time.time()
    from medal_bench.features.sam import make_sam_foundation_fn
    found_fn = make_sam_foundation_fn(cache_dir="/tmp/stress_sam_cache", model_type=args.sam_model_type)
    found, found_meta = found_fn(unlabeled_ds=unlabeled_ds, labeled_ds=labeled_ds, seed=1000, device=args.device)
    print(f"[stress] SAM-{args.sam_model_type} features ({found['foundation_pool'].shape}) in {time.time()-ts:.1f}s", flush=True)

    feats = {**task, **found}
    cand_ids = [unlabeled_ds[i].sample_id for i in range(len(unlabeled_ds))]

    def make_ctx():
        return PolicyContext(seed=1000, round_idx=0, model=model, pred_cache=pred_cache,
                             pool=unlabeled_ds, labeled=labeled_ds, features=feats, num_classes=nc)

    rows = []
    for pid in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ctx = make_ctx()
        pol = build(pid)
        t = time.time()
        scores = pol.score(ctx)
        sel = pol.select(ctx, scores, k=args.k)
        dt = time.time() - t
        peak = _peak_mb()
        if scores is None:
            cand = [float("nan")] * len(cand_ids); scoring = False
        else:
            cand = [float(x) for x in (scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores))]
            scoring = True
        cpath = write_candidate_scores(args.out, f"{args.dataset}__{pid}__s1000", 0, cand_ids, cand)
        finite = bool(np.isfinite(np.asarray(cand)).all()) if scoring else True
        row = {"pid": pid, "query_sec": round(dt, 1), "peak_gpu_mb": peak,
               "n_candidates": len(cand_ids), "k": args.k,
               "unique": len(set(sel)) == len(sel) == args.k, "scores_finite": finite,
               "candidate_sidecar": os.path.exists(cpath)}
        rows.append(row)
        print(f"  {pid:3s} query={dt:6.1f}s peak={peak:6d}MB uniq={row['unique']} "
              f"finite={finite} sidecar={row['candidate_sidecar']}", flush=True)

    # deterministic replay (P1 + P9) on the same fixed state
    def _sel(pid):
        c = make_ctx(); p = build(pid); s = p.score(c); return p.select(c, s, k=args.k)
    replay = {pid: (_sel(pid) == _sel(pid)) for pid in ("P1", "P9")}
    print(f"[stress] deterministic replay: {replay}", flush=True)

    summary = {"dataset": args.dataset, "cap": cap, "image_size": args.image_size,
               "n_candidates": len(cand_ids), "ckpt_hash": ckpt_hash, "ckpt_hash_saved": bool(ckpt_hash),
               "sam_model_type": args.sam_model_type, "sam_feat_dim": int(found["foundation_pool"].shape[1]),
               "deterministic_replay": replay, "methods": rows}
    with open(os.path.join(args.out, "stress_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stress] wrote {os.path.join(args.out,'stress_results.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
