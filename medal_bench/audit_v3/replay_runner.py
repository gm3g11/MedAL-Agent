"""v3 fair test-set evaluation pass: REPLAY v1 selected_ids with checkpoint saving
and val+test evaluation per round. Does NOT call any policy's score/select for
selection purposes — selections are read verbatim from v1 JSONLs.

For P6 PEAL and P9 PAAL, optional diagnostic logging is enabled but does NOT
influence the selection (selections are still from v1 JSONLs).

Exact v1 training config:
  nnU-Net 2D PlainConvUNet (features 32,64,128,256,320), dropout 0.1
  256x256 input, AdamW lr=1e-3, batch 8, 250 iters/round
  loss = mean(CE + (1 - foreground Dice))
  from-scratch each round
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.audit_v3.eval_full import eval_full
from medal_bench.data.base import MedALDataset, Sample
from medal_bench.models.nnunet import build_unet_2d
from medal_bench.runner.al_loop import _IndexedSubset, _build_model
from medal_bench.runner.seeds import seed_all
from medal_bench.runner.splits import SplitView, make_split
from medal_bench.runner.trainer import train_from_scratch
from medal_bench.runner.al_loop import TrainConfig as ALTrainConfig


PILOT_TRAIN = ALTrainConfig(
    num_iters=250, batch_size=8, lr=1e-3,
    image_size=256,
    features_per_stage=(32, 64, 128, 256, 320),
    dropout_p=0.1,
)
BUDGET_FRACS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


@dataclass
class ReplayConfig:
    dataset_name: str
    policy_id: str
    seed: int
    v1_jsonl: str        # path to v1 JSONL with selected_ids
    out_dir: str         # where the v3 JSONL is written
    ckpt_dir: str        # where per-round checkpoints are saved
    device: str = "cuda:0"
    save_intermediate_ckpts: bool = True
    only_rounds: Optional[list[int]] = None  # if set, only run these rounds (canary)


def _build_adapter(dataset_name: str):
    from medal_bench.data.adapters import (
        ISIC2018Adapter, CVCClinicDBAdapter, BUSIAdapter, PROMISE12Adapter,
    )
    DATA_ROOT = "/groups/echambe2/datasets/data"
    factories = {
        "isic2018":     lambda: ISIC2018Adapter(f"{DATA_ROOT}/2d/isic2018_task1", split="train"),
        "cvc_clinicdb": lambda: CVCClinicDBAdapter(f"{DATA_ROOT}/2d/cvc_clinicdb"),
        "busi":         lambda: BUSIAdapter(f"{DATA_ROOT}/2d/busi"),
        "promise12":    lambda: PROMISE12Adapter(f"{DATA_ROOT}/2d/promise12"),
    }
    return factories[dataset_name]()


def _cumulative_budget(pool_size: int) -> list[int]:
    import math as _math
    plan, last = [], -1
    for f in BUDGET_FRACS:
        n = max(1, int(_math.ceil(f * pool_size)))
        n = min(n, pool_size)
        n = max(n, last + 1)
        plan.append(n); last = n
    return plan


def _load_v1_selected_ids_per_round(jsonl_path: str) -> list[list[str]]:
    """Read v1 JSONL → list of selected_ids per round (length = #rounds; last round usually empty)."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            out.append(list(r.get("selected_ids", [])))
    return out


def _cold_start_indices(n_pool: int, n_init: int, seed: int) -> list[int]:
    """Reproduce v1's cold-start: shuffle range(n_pool) with RandomState(seed); take first n_init."""
    rng = np.random.RandomState(seed)
    pool_idx = list(range(n_pool))
    rng.shuffle(pool_idx)
    return list(pool_idx[:n_init])


def run_replay(cfg: ReplayConfig) -> list[dict]:
    seed_all(cfg.seed)
    out_path = Path(cfg.out_dir) / f"{cfg.dataset_name}__{cfg.policy_id}__s{cfg.seed}.jsonl"
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Build adapter + splits — IDENTICAL to v1
    adapter = _build_adapter(cfg.dataset_name)
    split = make_split(adapter, seed=cfg.seed)
    train_view = SplitView(adapter, split.train, "train")
    val_view   = SplitView(adapter, split.val,   "val")
    test_view  = SplitView(adapter, split.test,  "test")

    # No pool_cap for any of our 4 datasets in v1 PILOT (only MSD07 was capped, and it's dropped).
    train_indices = list(range(len(train_view)))
    pool_subset = _IndexedSubset(train_view, train_indices, PILOT_TRAIN.image_size)
    val_subset  = _IndexedSubset(val_view,   list(range(len(val_view))),  PILOT_TRAIN.image_size)
    test_subset = _IndexedSubset(test_view,  list(range(len(test_view))), PILOT_TRAIN.image_size)

    # Cold start (round 0 labeled set)
    pool_size = len(pool_subset)
    plan = _cumulative_budget(pool_size)
    n_init = plan[0]
    cold_start_local_indices = _cold_start_indices(pool_size, n_init, cfg.seed)
    # Map local indices → sample_ids for v1-compat tracking
    sample_ids_in_pool = [pool_subset[i].sample_id for i in range(pool_size)]
    id_to_local_idx = {sid: i for i, sid in enumerate(sample_ids_in_pool)}
    cold_start_sample_ids = [sample_ids_in_pool[i] for i in cold_start_local_indices]

    # Load v1 selected_ids per round (rounds 0..R-1; last round has empty selected_ids)
    v1_selected_per_round = _load_v1_selected_ids_per_round(cfg.v1_jsonl)
    R = len(v1_selected_per_round)
    if R != 6:
        print(f"[warn] v1 trajectory has {R} rounds (expected 6); will replay all of them")

    # Determine which rounds to run
    rounds_to_run = list(range(R)) if cfg.only_rounds is None else cfg.only_rounds

    # Channels + num_classes
    first_img = pool_subset[0].image
    input_channels = int(first_img.shape[0])
    num_classes = adapter.num_classes

    records = []
    run_id = f"{cfg.dataset_name}__{cfg.policy_id}__s{cfg.seed}__v3"

    # Cumulative labeled set rebuilt round-by-round
    labeled_sample_ids_so_far = set(cold_start_sample_ids)

    for r in range(R):
        t_round_start = time.time()
        # FOR ROUND r, the labeled set is cold_start ∪ selected_ids[0..r-1]
        if r > 0:
            labeled_sample_ids_so_far.update(v1_selected_per_round[r-1])
        # Map sample_ids → pool indices
        missing = [sid for sid in labeled_sample_ids_so_far if sid not in id_to_local_idx]
        if missing:
            raise RuntimeError(f"r={r}: {len(missing)} v1 selected_ids not in current pool. Examples: {missing[:3]}")
        labeled_local = sorted({id_to_local_idx[sid] for sid in labeled_sample_ids_so_far})
        labeled_ds = _IndexedSubset(pool_subset, labeled_local, PILOT_TRAIN.image_size)

        if r not in rounds_to_run:
            print(f"[round {r}] skip (only_rounds={cfg.only_rounds})")
            continue

        # 1. Train from scratch (exact v1 config)
        t_train_start = time.time()
        model = _build_model(input_channels, num_classes, PILOT_TRAIN).to(cfg.device)
        train_stats = train_from_scratch(
            model, labeled_ds,
            num_iters=PILOT_TRAIN.num_iters, batch_size=PILOT_TRAIN.batch_size,
            lr=PILOT_TRAIN.lr, image_size=PILOT_TRAIN.image_size,
            num_classes=num_classes, device=cfg.device, seed=cfg.seed + r,
        )
        t_train = time.time() - t_train_start

        # 2. Save checkpoint (per round)
        if cfg.save_intermediate_ckpts:
            ckpt_path = Path(cfg.ckpt_dir) / f"{cfg.dataset_name}__{cfg.policy_id}__s{cfg.seed}__r{r}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "input_channels": input_channels,
                "num_classes": num_classes,
                "features_per_stage": PILOT_TRAIN.features_per_stage,
                "dropout_p": PILOT_TRAIN.dropout_p,
                "round": r,
                "n_labeled": len(labeled_local),
                "dataset": cfg.dataset_name,
                "policy_id": cfg.policy_id,
                "seed": cfg.seed,
                "v1_jsonl_origin": cfg.v1_jsonl,
            }, ckpt_path)
        else:
            ckpt_path = None

        # 3. Evaluate VAL and TEST
        t_val_start = time.time()
        val_metrics = eval_full(model, val_subset,
                                num_classes=num_classes,
                                image_size=PILOT_TRAIN.image_size,
                                device=cfg.device)
        t_val = time.time() - t_val_start
        t_test_start = time.time()
        test_metrics = eval_full(model, test_subset,
                                 num_classes=num_classes,
                                 image_size=PILOT_TRAIN.image_size,
                                 device=cfg.device)
        t_test = time.time() - t_test_start

        record = {
            "schema_version": "v3.0",
            "run_id": run_id,
            "dataset": cfg.dataset_name,
            "policy_id": cfg.policy_id,
            "seed": cfg.seed,
            "round": r,
            "labeled_count": len(labeled_local),
            "labeled_ratio": len(labeled_local) / pool_size,
            "ckpt_path": str(ckpt_path) if ckpt_path else None,
            "v1_jsonl_origin": cfg.v1_jsonl,
            "training": {
                "num_iters": PILOT_TRAIN.num_iters,
                "batch_size": PILOT_TRAIN.batch_size,
                "lr": PILOT_TRAIN.lr,
                "image_size": PILOT_TRAIN.image_size,
                **train_stats,
            },
            "metrics_val":  val_metrics,
            "metrics_test": test_metrics,
            "runtime_sec": {"train": t_train, "eval_val": t_val, "eval_test": t_test,
                            "total_round": time.time() - t_round_start},
        }
        records.append(record)
        # Append to JSONL atomically
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        print(f"[round {r}] labeled={len(labeled_local)} val_DSC={val_metrics['mean_dsc_fg']:.4f} "
              f"test_DSC={test_metrics['mean_dsc_fg']:.4f} val_HD95={val_metrics['mean_hd95_filtered_fg']:.2f} "
              f"test_HD95={test_metrics['mean_hd95_filtered_fg']:.2f} wall={time.time()-t_round_start:.1f}s "
              f"ckpt={'saved' if ckpt_path else 'skipped'}")

    return records
