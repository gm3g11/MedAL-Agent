"""SAM-B vs SAM-H pilot for P7/P8 (task 2).

Isolates the foundation-feature effect: same dataset split, same fixed initial
labeled set, same query policies — only the SAM model type changes. Reports
selected-ID overlap, extraction runtime, peak GPU memory, feature dim, and
feature-cache size. No training (P7/P8 use only frozen features), so differences
are purely due to SAM-B vs SAM-H features.

Usage:
    python -m medal_bench.runner.pilot_sam --out runs/pilot_sam --cap 160 --k 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import medal_bench.features.sam as sammod
from medal_bench.features.sam import extract_sam_features, resolve_sam_spec, _cache_path, SamPreprocessConfig
from medal_bench.data.adapters import DATASET_REGISTRY
from medal_bench.runner.splits import make_split, SplitView
from medal_bench.policies import build, PolicyContext

DATA_ROOT = "/groups/echambe2/datasets/data"
DATASETS = ["busi", "cvc_clinicdb", "isic2018", "promise12"]


def _free_gpu():
    sammod._LOADED.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pilot_sam")
    ap.add_argument("--cache", default="/tmp/pilot_sam_cache")
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # prepare per-dataset capped pool + labeled views once
    views = {}
    for ds in DATASETS:
        adapter = DATASET_REGISTRY[ds](DATA_ROOT)
        split = make_split(adapter, seed=args.seed)
        train = split.train[: args.cap + 8]
        pool_ds = SplitView(adapter, train[: args.cap], "pool")
        lab_ds = SplitView(adapter, train[args.cap: args.cap + 8], "lab")
        views[ds] = (adapter, pool_ds, lab_ds)

    results = {}
    for mt in ("vit_b", "vit_h"):
        _free_gpu()
        spec = resolve_sam_spec(mt)
        for ds in DATASETS:
            adapter, pool_ds, lab_ds = views[ds]
            t0 = time.time()
            try:
                pf = extract_sam_features(pool_ds, cache_dir=args.cache, spec=spec, device=args.device)
                lf = extract_sam_features(lab_ds, cache_dir=args.cache, spec=spec, device=args.device)
            except Exception as e:  # fail clearly, do NOT fall back to SAM-B
                results[(mt, ds)] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[{mt}/{ds}] FAIL: {e}")
                continue
            dt = time.time() - t0
            peak_mb = int(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else -1
            cache_mb = round(os.path.getsize(_cache_path(args.cache, pool_ds.name, spec, SamPreprocessConfig())) / 1e6, 2)
            ctx = PolicyContext(seed=args.seed, round_idx=0, num_classes=adapter.num_classes,
                                features={"foundation_pool": pf, "foundation_label": lf})
            sel_p7 = build("P7").select(ctx, None, k=args.k)
            sel_p8 = build("P8").select(ctx, None, k=args.k)
            ids = pool_ds.sample_ids()
            results[(mt, ds)] = {
                "extract_sec": round(dt, 1), "peak_gpu_mb": peak_mb, "feat_dim": int(pf.shape[1]),
                "n_pool": int(pf.shape[0]), "cache_mb": cache_mb,
                "p7_ids": [ids[i] for i in sel_p7], "p8_ids": [ids[i] for i in sel_p8],
            }
            print(f"[{mt}/{ds}] {dt:.1f}s peak={peak_mb}MB dim={pf.shape[1]} cache={cache_mb}MB")

    # overlap report
    print("\n=== SAM-B vs SAM-H pilot ===")
    print(f"{'dataset':14s} {'dim':>4s} {'t_b':>6s} {'t_h':>6s} {'mem_b':>6s} {'mem_h':>6s} "
          f"{'P7overlap':>9s} {'P8overlap':>9s}")
    rows = []
    for ds in DATASETS:
        b, h = results.get(("vit_b", ds), {}), results.get(("vit_h", ds), {})
        if "error" in h or "error" in b or not b or not h:
            print(f"{ds:14s}  ERROR b={b.get('error','')} h={h.get('error','')}")
            rows.append({"dataset": ds, "error_b": b.get("error"), "error_h": h.get("error")})
            continue
        ov7 = len(set(b["p7_ids"]) & set(h["p7_ids"])) / max(1, len(b["p7_ids"]))
        ov8 = len(set(b["p8_ids"]) & set(h["p8_ids"])) / max(1, len(b["p8_ids"]))
        print(f"{ds:14s} {h['feat_dim']:4d} {b['extract_sec']:6.1f} {h['extract_sec']:6.1f} "
              f"{b['peak_gpu_mb']:6d} {h['peak_gpu_mb']:6d} {ov7:9.2f} {ov8:9.2f}")
        rows.append({"dataset": ds, "feat_dim": h["feat_dim"], "n_pool": h["n_pool"],
                     "t_b": b["extract_sec"], "t_h": h["extract_sec"],
                     "mem_b": b["peak_gpu_mb"], "mem_h": h["peak_gpu_mb"],
                     "cache_b_mb": b["cache_mb"], "cache_h_mb": h["cache_mb"],
                     "p7_overlap": round(ov7, 3), "p8_overlap": round(ov8, 3)})
    with open(os.path.join(args.out, "pilot_sam_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'pilot_sam_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
