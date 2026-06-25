"""CLI: replay a single (dataset, policy, seed) v1 cell, save checkpoints + val/test metrics."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = "/groups/echambe2/gmeng/MedAL-Agent/repo/code"
DEFAULT_V1_DIR = f"{REPO}/runs/pilot_v1"
DEFAULT_OUT_DIR = f"{REPO}/runs/test_eval_v3"
DEFAULT_CKPT_DIR = f"{REPO}/runs/test_eval_v3/ckpts"
DEFAULT_HF_CACHE = "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["busi","cvc_clinicdb","isic2018","promise12"])
    ap.add_argument("--policy", required=True, choices=[f"P{i}" for i in range(10)])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--v1-dir", default=DEFAULT_V1_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only-rounds", default=None, help="comma-list of rounds, e.g. '0,5' for canary")
    ap.add_argument("--no-intermediate-ckpts", action="store_true",
                    help="only save final-round ckpt (saves disk)")
    args = ap.parse_args(argv)

    os.environ["HF_HOME"] = DEFAULT_HF_CACHE
    os.environ["HF_HUB_CACHE"] = DEFAULT_HF_CACHE

    v1_jsonl = f"{args.v1_dir}/{args.dataset}__{args.policy}__s{args.seed}.jsonl"
    if not os.path.exists(v1_jsonl):
        print(f"[run_one_replay] FAIL: v1 trajectory missing: {v1_jsonl}", file=sys.stderr)
        return 2

    only_rounds = None
    if args.only_rounds:
        only_rounds = [int(x) for x in args.only_rounds.split(",")]

    from medal_bench.audit_v3.replay_runner import ReplayConfig, run_replay
    cfg = ReplayConfig(
        dataset_name=args.dataset, policy_id=args.policy, seed=args.seed,
        v1_jsonl=v1_jsonl, out_dir=args.out_dir, ckpt_dir=args.ckpt_dir,
        device=args.device,
        save_intermediate_ckpts=not args.no_intermediate_ckpts,
        only_rounds=only_rounds,
    )
    print(f"[run_one_replay] {args.dataset}/{args.policy}/seed={args.seed} v1={v1_jsonl}")

    try:
        recs = run_replay(cfg)
        last = recs[-1] if recs else None
        if last:
            print(f"[run_one_replay] DONE rounds={len(recs)} last_val_DSC={last['metrics_val']['mean_dsc_fg']:.4f} "
                  f"last_test_DSC={last['metrics_test']['mean_dsc_fg']:.4f}")
        return 0
    except Exception as e:
        print(f"[run_one_replay] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
