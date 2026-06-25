"""Round-driver for active learning.

One call to ``run_al`` runs a complete experiment for one
``(policy, dataset, seed)`` tuple, emitting one TrajectoryRecord per round
to a JSONL file.

Round 0:
  - random initial labeled set of size budget_plan[0]
  - train from scratch
  - eval on val
  - log

Rounds 1..R-1:
  - build pred_cache + task features + foundation features (smoke stub)
  - policy.score + policy.select(k = delta-to-next-checkpoint)
  - reveal labels (move selected pool indices into labeled set)
  - retrain from scratch
  - eval on val
  - log

Constraints honored:
  - Patient-grouped splits if adapter.patient_ids() is non-None.
  - Policies receive only pool + labeled metadata, never val/test datasets.
  - Foundation block in the trajectory records encoder_id + cache_version.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from medal_bench.data.base import MedALDataset, Sample
from medal_bench.models.nnunet import build_unet_2d
from medal_bench.policies import build, PolicyContext
from medal_bench.runner.eval import eval_segmentation
from medal_bench.runner.feature_extractor import extract_task_unet_features
from medal_bench.runner.foundation_stub import (
    extract_foundation_features_stub, foundation_stub_meta,
)
from medal_bench.runner.prediction_cache import (
    build_prediction_cache, stream_pool_reduce, PredictionCache,
)


def _default_foundation_fn(*, unlabeled_ds, labeled_ds, seed, device):
    """Backward-compat default: the seeded-random stub. Pilot runs pass
    cfg.foundation_features_fn=make_sam_features_fn() instead."""
    return (
        {
            "foundation_pool":  extract_foundation_features_stub(unlabeled_ds, seed=seed),
            "foundation_label": extract_foundation_features_stub(labeled_ds,   seed=seed),
        },
        foundation_stub_meta(),
    )
from medal_bench.runner.seeds import seed_all, seed_torch, component_seeds
from medal_bench.runner.splits import make_split, SplitView
from medal_bench.runner.trainer import train_from_scratch
from medal_bench.runner.trajectory import (
    TrajectoryRecord, append_record, state_dict_hash, config_hash,
    write_candidate_scores, write_predictions,
)


def _sanitize_diagnostics(d: dict) -> dict:
    """Drop torch tensors / numpy arrays from diagnostics; replace with shape +
    mean so the trajectory record stays JSON-serializable."""
    import numpy as _np
    out = {}
    dropped = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            dropped.append(k)
            out[f"{k}__mean"] = float(v.float().mean().detach().cpu())
            out[f"{k}__shape"] = list(v.shape)
        elif isinstance(v, _np.ndarray):
            dropped.append(k)
            out[f"{k}__mean"] = float(v.mean())
            out[f"{k}__shape"] = list(v.shape)
        else:
            out[k] = v
    if dropped:
        out["_dropped_tensor_keys"] = dropped
    return out


# Sample wrapper that exposes a single chosen sample id <-> image generator
def _iter_images(ds: MedALDataset, indices: list[int], image_size: int):
    """Yield (sample_id, resized image tensor) for build_prediction_cache."""
    import torch.nn.functional as F
    for i in indices:
        s = ds[i]
        x = torch.from_numpy(s.image).unsqueeze(0)
        x = F.interpolate(x, size=(image_size, image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        yield s.sample_id, x


class _IndexedSubset(MedALDataset):
    """In-memory subset of an adapter at given indices, resized to image_size.

    Built per-round so the pred_cache / feature pass operates on canonicalized
    inputs without re-resizing in three different places.
    """
    PREPROC_VERSION = "v3-letterbox-validbbox"  # fp16 image + int16 mask + valid_bbox on disk

    def __init__(self, base: MedALDataset, indices: list[int], image_size: int,
                 aspect_preserve: bool = False, cache_dir: Optional[str] = None):
        from medal_bench.runner.trainer import _resize_image, _resize_mask, valid_bbox
        import numpy as _np
        self._base = base
        self._idx = list(indices)
        self.name = f"{base.name}_subset"
        self.modality = base.modality
        self.target = base.target
        self.dim = base.dim
        self.query_unit = base.query_unit
        self.num_classes = base.num_classes
        self.cache_status = "disabled"

        # ---- B6 preprocessed-array disk cache (keyed by content) ----
        # hoist sample_ids() out of the comprehension: it rebuilds an O(n) list, so
        # calling it per-index is O(n^2) and HANGS on large-val datasets (e.g. the
        # 11k-slice ext_abdoment1k val set). One call + indexed lookups = O(n), same ids.
        _base_sids = base.sample_ids()
        ids = [_base_sids[i] for i in self._idx]
        if cache_dir:
            import hashlib
            key = hashlib.sha256(
                f"{base.name}|{image_size}|{aspect_preserve}|{self.PREPROC_VERSION}|"
                f"{','.join(ids)}".encode()).hexdigest()[:16]
            cpath = os.path.join(cache_dir, f"{base.name}__sz{image_size}__{key}.npz")
            if os.path.exists(cpath):
                z = _np.load(cpath, allow_pickle=True)
                imgs, masks = z["images"], z["masks"]   # fp16 / int16 on disk
                pids = z["patient_ids"]; slis = z["slice_indices"]; sids = z["sample_ids"]
                vbs = z["valid_bboxes"] if "valid_bboxes" in z.files else None
                self._samples = [
                    Sample(sample_id=str(sids[k]),
                           image=imgs[k].astype(_np.float32), mask=masks[k].astype(_np.int64),
                           meta=({"valid_bbox": tuple(int(x) for x in vbs[k])} if vbs is not None else {}),
                           patient_id=(None if pids[k] == "" else str(pids[k])),
                           slice_index=(None if int(slis[k]) < 0 else int(slis[k])))
                    for k in range(len(sids))
                ]
                self.cache_status = "hit"
                return
            self.cache_status = "miss"
        # For 3D-source adapters, base.patient_ids() groups slices by volume.
        # We pre-load in (patient_id, slice_index) order so the adapter's volume
        # LRU sees each volume exactly once instead of thrashing — critical for
        # MSD07 / PROMISE12 where random index order with LRU(maxsize=4) caused
        # ~22500 NIfTI loads instead of ~225. We hold the un-shuffled output
        # mapped back to the requested index order.
        base_pids = base.patient_ids() if hasattr(base, "patient_ids") else None
        if base_pids is not None:
            load_order = sorted(
                range(len(self._idx)),
                key=lambda j: (base_pids[self._idx[j]],
                               getattr(base[self._idx[j]] if False else None, "slice_index", 0) or 0),
            )
            # cheaper key: avoid building Sample just for sort; use a small probe
            load_order = sorted(
                range(len(self._idx)),
                key=lambda j: (base_pids[self._idx[j]], self._idx[j]),
            )
        else:
            load_order = list(range(len(self._idx)))
        # eager resize so __getitem__ is constant-time and policies see uniform shape
        self._samples: list[Sample] = [None] * len(self._idx)  # type: ignore[list-item]
        for j in load_order:
            i = self._idx[j]
            s = base[i]
            oh, ow = int(s.image.shape[-2]), int(s.image.shape[-1])
            img = _resize_image(s.image, image_size, aspect_preserve).numpy()
            mask = _resize_mask(s.mask, image_size, aspect_preserve).numpy().astype(_np.int64)
            # valid (un-padded) rect of the resized canvas. REUSE a base sample's
            # existing bbox (nested subsets re-resize an already-square image and
            # would otherwise recompute a wrong full-canvas bbox); else compute from
            # the original H,W.
            vb = s.meta.get("valid_bbox") if isinstance(s.meta, dict) else None
            if vb is None:
                vb = valid_bbox(oh, ow, image_size, aspect_preserve)
            meta = dict(s.meta) if isinstance(s.meta, dict) else {}
            meta["valid_bbox"] = tuple(int(x) for x in vb)
            self._samples[j] = Sample(
                sample_id=s.sample_id, image=img, mask=mask,
                meta=meta, patient_id=s.patient_id, slice_index=s.slice_index,
            )
        if cache_dir and self.cache_status == "miss":
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cpath + ".tmp.npz"
            _np.savez(
                tmp,
                images=_np.stack([s.image for s in self._samples]).astype(_np.float16),
                masks=_np.stack([s.mask for s in self._samples]).astype(_np.int16),
                sample_ids=_np.array([s.sample_id for s in self._samples]),
                patient_ids=_np.array([s.patient_id or "" for s in self._samples]),
                slice_indices=_np.array([-1 if s.slice_index is None else s.slice_index
                                         for s in self._samples]),
                valid_bboxes=_np.array([s.meta["valid_bbox"] for s in self._samples],
                                       dtype=_np.int32),
            )
            os.replace(tmp, cpath)

    def __len__(self) -> int: return len(self._samples)
    def sample_ids(self) -> list[str]: return [s.sample_id for s in self._samples]
    def __getitem__(self, i: int) -> Sample: return self._samples[i]
    def patient_ids(self) -> Optional[list[str]]:
        # per-sample membership: return the real list whenever ANY sample is
        # grouped (not gated on sample[0] alone, which dropped the whole list to
        # None if the first sample happened to be ungrouped).
        if self._samples and any(s.patient_id for s in self._samples):
            return [s.patient_id for s in self._samples]
        return None


def _valid_bboxes_for(ds: "MedALDataset", image_size: int) -> Optional[np.ndarray]:
    """(N, 4) int array of per-sample valid rectangles (y0, x0, h, w) from each
    sample's meta['valid_bbox']. Returns None when every sample is full-canvas
    (no padding), so policies fall back to the whole canvas with zero overhead."""
    bboxes = []
    any_pad = False
    for i in range(len(ds)):
        meta = ds[i].meta
        vb = meta.get("valid_bbox") if isinstance(meta, dict) else None
        if vb is None:
            vb = (0, 0, image_size, image_size)
        y0, x0, h, w = (int(v) for v in vb)
        if not (y0 == 0 and x0 == 0 and h == image_size and w == image_size):
            any_pad = True
        bboxes.append((y0, x0, h, w))
    if not any_pad:
        return None
    return np.asarray(bboxes, dtype=np.int64)


@dataclass
class TrainConfig:
    num_iters: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    image_size: int = 256
    features_per_stage: tuple = (16, 32, 64)
    dropout_p: float = 0.1
    aspect_preserve: bool = False   # True = adaptive long-side letterbox to image_size
    # frozen_v4 adaptive training: train each round to a train-loss plateau instead of a
    # fixed num_iters (removes under-fitting bias for difficulty-based AL, scales without
    # per-dataset tuning). OFF by default => frozen_v3 fixed-iter behaviour is unchanged.
    # Caps are PROVISIONAL pending the iters-probe calibration.
    adaptive_iters: bool = False
    min_iters: int = 500
    max_iters: int = 3000
    plateau_window: int = 100
    plateau_patience: int = 5
    plateau_min_delta: float = 0.003   # ABSOLUTE smoothed-loss delta floor
    plateau_rel_delta: float = 0.0     # relative term: threshold = max(abs, rel*|best_loss|)


@dataclass
class RunConfig:
    policy_id: str
    policy_config: dict
    dataset_name: str               # echoed into the trajectory record
    seed: int
    budget_plan: list[int]          # cumulative #-labeled at checkpoints (#0 = initial)
    train: TrainConfig
    out_jsonl: str
    device: str = "cuda:0"
    # max pool size to use (for smoke; None = full)
    pool_cap: Optional[int] = None
    val_cap: Optional[int] = None
    # foundation feature factory; default = seeded-random stub
    foundation_features_fn: Optional[object] = None
    # HD95 + ASD eval at the FINAL round (slow). Smoke runs leave False.
    compute_surface_metrics_at_final: bool = False
    # explicit set of round indices to compute HD95/ASD on (first/mid/final
    # policy for Stage 1). If non-None, overrides compute_surface_metrics_at_final.
    surface_rounds: Optional[set] = None
    # preprocessed-array disk cache dir (B6). None = eager rebuild each run.
    preproc_cache_dir: Optional[str] = None
    # save the per-round model state_dict to disk (ckpt_path logged). The
    # ckpt_hash is ALWAYS logged regardless; this just persists weights too.
    save_checkpoints: bool = False
    # frozen_v3 prediction saving: dump compressed val masks + ids + valid masks
    # every round (always-on for Stage 2). save_logits additionally stores fp16
    # softmax probs (heavier; gated by the canary storage estimate before Wave 2).
    save_predictions: bool = False
    save_logits: bool = False
    # When pool_cap applies AND True, pre-scan the train pool's masks and
    # cap with fg_ratio% foreground-containing slices + (1-fg_ratio)% bg.
    # Critical for MSD07 / sparse-FG datasets where random sub-sampling
    # starves the labeled set of foreground.
    stratify_pool_by_fg: bool = False
    stratify_fg_ratio: float = 0.5

    def __post_init__(self):
        if self.foundation_features_fn is None:
            self.foundation_features_fn = _default_foundation_fn


def _stratified_pool_cap(adapter, indices, target_size, rng, fg_ratio: float = 0.5):
    """Cap pool indices preferring slices that contain any foreground.

    Pre-scans every index's mask through the adapter. For datasets with a
    patient-grouped index, scan iterates in (patient_id, base_index) order so
    the adapter's volume LRU sees each volume exactly once. Returns indices
    in the original ``indices`` order (a subset of it)."""
    if len(indices) <= target_size:
        return list(indices)
    base_pids = adapter.patient_ids() if hasattr(adapter, "patient_ids") else None
    if base_pids is not None:
        scan_order = sorted(indices, key=lambda i: (base_pids[i], i))
    else:
        scan_order = list(indices)
    # I/O-latency-bound: bridged datasets are pre-sliced 2D PNGs on NFS, so this scan
    # is ~Nslices tiny reads (ext_abdoment1k ~130k -> ~hours SEQUENTIALLY at NFS latency).
    # Parallelize with THREADS to hide that latency. ThreadPoolExecutor.map preserves
    # input order, so fg_pos/fg_neg are built in the SAME scan_order as the sequential
    # loop -> rng.choice below picks the IDENTICAL slices (pool selection unchanged /
    # deterministic). The patient-grouped scan_order + thread-safe volume LRU still load
    # each native-3D volume once.
    from concurrent.futures import ThreadPoolExecutor
    def _slice_has_fg(i: int) -> bool:
        m = adapter[i].mask
        return bool((m > 0).any()) if m is not None else False
    fg_pos: list[int] = []
    fg_neg: list[int] = []
    with ThreadPoolExecutor(max_workers=24) as _ex:
        for i, has in zip(scan_order, _ex.map(_slice_has_fg, scan_order)):
            (fg_pos if has else fg_neg).append(i)
    n_fg_target = int(round(target_size * fg_ratio))
    n_fg_take = min(len(fg_pos), n_fg_target)
    n_bg_take = min(target_size - n_fg_take, len(fg_neg))
    chosen_fg = rng.choice(fg_pos, size=n_fg_take, replace=False).tolist() if n_fg_take > 0 else []
    chosen_bg = rng.choice(fg_neg, size=n_bg_take, replace=False).tolist() if n_bg_take > 0 else []
    selected = set(int(x) for x in chosen_fg + chosen_bg)
    return [i for i in indices if i in selected]


def _build_model(input_channels: int, num_classes: int, train_cfg: TrainConfig) -> torch.nn.Module:
    return build_unet_2d(
        input_channels=input_channels,
        num_classes=num_classes,
        features_per_stage=train_cfg.features_per_stage,
        dropout_p=train_cfg.dropout_p,
    )


def _initial_labeled_path(cfg: "RunConfig", dataset_name: str) -> str:
    """Shared (per dataset+seed+init-size) location for the round-0 labeled set,
    so EVERY policy at the same (dataset, seed, profile) reuses the identical
    initial set. Lives next to the trajectory JSONLs under ``init_sets/``."""
    base = os.path.dirname(os.path.abspath(cfg.out_jsonl)) or "."
    n_init = cfg.budget_plan[0]
    return os.path.join(base, "init_sets", f"{dataset_name}__s{cfg.seed}__n{n_init}.json")


def _load_or_make_initial_labeled(cfg: "RunConfig", dataset_name: str,
                                  pool_subset, rng, n_init: int) -> list[str]:
    """Return the round-0 labeled SAMPLE IDs. Loads the saved set if present
    (fairness across policies, robust to RNG-order changes); else draws it from
    the seeded RNG and persists it atomically. Note: ``rng`` is not used after
    this in run_al, so skipping the shuffle on the load path is safe."""
    path = _initial_labeled_path(cfg, dataset_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["sample_ids"]
    pool_idx = list(range(len(pool_subset)))
    rng.shuffle(pool_idx)
    ids = sorted(pool_subset[i].sample_id for i in pool_idx[:n_init])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"dataset": dataset_name, "seed": cfg.seed,
                   "n_init": n_init, "sample_ids": ids}, f)
    os.replace(tmp, path)
    return ids


def _load_or_make_pool_indices(cfg: "RunConfig", dataset_name: str, train_view) -> list[int]:
    """Return the (capped + fg-stratified) train-pool indices, CACHED per
    (dataset, seed, pool_cap, fg_ratio). The fg-stratify pre-scan loads every
    train mask (e.g. 21k NIfTI slices for MSD07) — caching it means the scan runs
    ONCE per (dataset, seed) instead of once per (policy x seed). Uses a dedicated
    RNG so the main RNG (val_cap + init set) is unaffected by cache hit/miss."""
    n = len(train_view)
    if cfg.pool_cap is None or n <= cfg.pool_cap:
        return list(range(n))
    # GLOBAL cache (keyed by dataset+seed+pool_cap+fg_ratio): the expensive fg-scan
    # runs ONCE EVER and is reused across every out-dir / wave / smoke (not per-run).
    base = os.environ.get("MEDAL_AL_STATE", "/groups/echambe2/gmeng/MedAL-Agent/cache/al_state")
    tag = f"cap{cfg.pool_cap}" + (f"_fg{cfg.stratify_fg_ratio}" if cfg.stratify_pool_by_fg else "")
    path = os.path.join(base, "pool_sets", f"{dataset_name}__s{cfg.seed}__{tag}.json")
    sids = train_view.sample_ids()
    if os.path.exists(path):
        sel = json.load(open(path))["sample_ids"]
        pos = {sid: i for i, sid in enumerate(sids)}
        return [pos[sid] for sid in sel if sid in pos]
    cap_rng = np.random.RandomState(cfg.seed + 7)   # dedicated; does not touch main rng
    idx = list(range(n))
    if cfg.stratify_pool_by_fg:
        idx = _stratified_pool_cap(train_view, idx, cfg.pool_cap, cap_rng, fg_ratio=cfg.stratify_fg_ratio)
    else:
        idx = cap_rng.choice(idx, size=cfg.pool_cap, replace=False).tolist()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump({"dataset": dataset_name, "seed": cfg.seed, "pool_cap": cfg.pool_cap,
               "sample_ids": [sids[i] for i in idx]}, open(tmp, "w"))
    os.replace(tmp, path)
    return idx


def run_al(adapter: MedALDataset, cfg: RunConfig) -> list[dict]:
    """Run one AL experiment, returning the list of trajectory record dicts."""
    seed_all(cfg.seed)
    split = make_split(adapter, seed=cfg.seed)

    # Build per-split views
    train_view = SplitView(adapter, split.train, "train")
    val_view = SplitView(adapter, split.val, "val")

    # Cap the train pool (cached per dataset+seed; fg-stratify scan runs once)
    rng = np.random.RandomState(cfg.seed)
    train_indices = _load_or_make_pool_indices(cfg, adapter.name, train_view)
    val_indices = list(range(len(val_view)))
    if cfg.val_cap is not None and len(val_indices) > cfg.val_cap:
        val_indices = rng.choice(val_indices, size=cfg.val_cap, replace=False).tolist()

    # eager-resized subsets the policies + cache will read (B6 disk cache on
    # the first-level pool/val subsets; labeled/unlabeled derive in-memory).
    pool_subset = _IndexedSubset(train_view, train_indices, cfg.train.image_size,
                                 cfg.train.aspect_preserve, cache_dir=cfg.preproc_cache_dir)
    val_subset = _IndexedSubset(val_view, val_indices, cfg.train.image_size,
                                cfg.train.aspect_preserve, cache_dir=cfg.preproc_cache_dir)
    _cache_status = {"pool": pool_subset.cache_status, "val": val_subset.cache_status}

    # budget-denominator provenance (frozen_v3 / M2): pool_subset IS the realized
    # post-cap/fg-stratified pool, so len(pool_subset) is the TRUE AL_pool_N. Logged
    # every round so fractions are auditable against both the pool and full train set.
    _actual_AL_pool_N = len(pool_subset)
    _full_train_N = len(train_view)
    _total_rounds = len(cfg.budget_plan)
    _budget_denominator = {
        "full_train_N": _full_train_N,
        "requested_pool_cap": cfg.pool_cap,
        "actual_AL_pool_N": _actual_AL_pool_N,
        "budget_plan": list(cfg.budget_plan),
        "fraction_of_AL_pool": cfg.budget_plan[-1] / max(1, _actual_AL_pool_N),
        "fraction_of_full_train": cfg.budget_plan[-1] / max(1, _full_train_N),
    }

    # First sample tells us input_channels
    first_img = pool_subset[0].image
    input_channels = int(first_img.shape[0])
    num_classes = adapter.num_classes

    # Initial labeled set: random, but persisted by (dataset, seed, n_init) and
    # reused across ALL policies so the cold start is provably identical.
    n_init = cfg.budget_plan[0]
    init_ids = _load_or_make_initial_labeled(cfg, adapter.name, pool_subset, rng, n_init)
    id_to_local = {pool_subset[i].sample_id: i for i in range(len(pool_subset))}
    _missing = [sid for sid in init_ids if sid not in id_to_local]
    assert not _missing, f"initial-set sample_ids absent from pool: {_missing[:3]}"
    labeled_local = set(id_to_local[sid] for sid in init_ids)
    assert len(labeled_local) == n_init, \
        f"initial set size {len(labeled_local)} != budget_plan[0]={n_init}"

    records: list[dict] = []
    run_id = f"{adapter.name}__{cfg.policy_id}__s{cfg.seed}"

    # Build the policy ONCE and reuse it across all rounds so policies that
    # PERSIST internal state across rounds (e.g. PAAL's Accuracy Predictor)
    # actually see continuity. Previously a fresh build() per round threw away
    # any cached state. NOTE: this is unrelated to the AL "cold start" — the
    # initial labeled set at round 0 is still a uniform-random subset shared
    # by every policy at the same seed (see pool_idx[:n_init] below).
    policy = build(cfg.policy_id, **cfg.policy_config)
    policy_class_name = type(policy).__name__
    init_path = _initial_labeled_path(cfg, adapter.name)
    cfg_hash = config_hash({
        "budget_plan": cfg.budget_plan,
        "num_iters": cfg.train.num_iters, "batch_size": cfg.train.batch_size,
        "lr": cfg.train.lr, "image_size": cfg.train.image_size,
        "features_per_stage": list(cfg.train.features_per_stage),
        "dropout_p": cfg.train.dropout_p,
        "pool_cap": cfg.pool_cap, "val_cap": cfg.val_cap,
        "stratify_pool_by_fg": cfg.stratify_pool_by_fg,
        "preprocess_version": "v1",
    })

    _cuda = isinstance(cfg.device, str) and cfg.device.startswith("cuda") and torch.cuda.is_available()
    # GPU identity (logged per cell); CUDA_VISIBLE_DEVICES isolates the alloc to index 0.
    if _cuda:
        try:
            _gpu_name = torch.cuda.get_device_name(0)
            _gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        except Exception:
            _gpu_name, _gpu_total_mb = "", None
    else:
        _gpu_name, _gpu_total_mb = "", None

    def _peak_gpu_mb():
        try:
            return torch.cuda.max_memory_allocated() / 1e6
        except Exception:
            return None
    for r, cum_target in enumerate(cfg.budget_plan):
        t0 = time.time()
        if _cuda:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # frozen_v3 component seeding (M3): anchor this round to seed+r so model
        # weight init, the batch loader, training dropout, and the query RNG are each
        # reproducible independent of prior rounds' op order. seed_all is the floor;
        # each component then consumes its own derived stream at its call site.
        round_seed = cfg.seed + r
        cseeds = component_seeds(round_seed)
        seed_all(round_seed)

        # Build labeled / unlabeled subsets at this round
        labeled_indices = sorted(labeled_local)
        unlabeled_indices = sorted(set(range(len(pool_subset))) - labeled_local)
        labeled_ds = _IndexedSubset(pool_subset, labeled_indices, cfg.train.image_size, cfg.train.aspect_preserve)
        unlabeled_ds = _IndexedSubset(pool_subset, unlabeled_indices, cfg.train.image_size, cfg.train.aspect_preserve)

        # 1. Train from scratch (model_init_seed anchors weight init; the loader and
        # dropout draw their own component streams inside train_from_scratch).
        seed_torch(cseeds["model_init_seed"])
        model = _build_model(input_channels, num_classes, cfg.train).to(cfg.device)
        train_stats = train_from_scratch(
            model, labeled_ds,
            num_iters=cfg.train.num_iters, batch_size=cfg.train.batch_size,
            lr=cfg.train.lr, image_size=cfg.train.image_size,
            num_classes=num_classes, device=cfg.device,
            seed=cseeds["loader_seed"], dropout_seed=cseeds["dropout_seed"],
            adaptive=cfg.train.adaptive_iters,
            min_iters=cfg.train.min_iters, max_iters=cfg.train.max_iters,
            plateau_window=cfg.train.plateau_window,
            plateau_patience=cfg.train.plateau_patience,
            plateau_min_delta=cfg.train.plateau_min_delta,
            plateau_rel_delta=cfg.train.plateau_rel_delta,
        )
        t_train = time.time() - t0

        # checkpoint provenance for the query-time model
        ckpt_hash = state_dict_hash(model)
        ckpt_path = ""
        if cfg.save_checkpoints:
            cdir = os.path.join(os.path.dirname(os.path.abspath(cfg.out_jsonl)), "ckpts")
            os.makedirs(cdir, exist_ok=True)
            ckpt_path = os.path.join(cdir, f"{run_id}__r{r}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # 2. Eval (surface metrics only on the final round, opt-in)
        t1 = time.time()
        is_final = (r + 1 == len(cfg.budget_plan))
        if cfg.surface_rounds is not None:
            do_surface = r in cfg.surface_rounds
        else:
            do_surface = cfg.compute_surface_metrics_at_final and is_final
        # I/O: only the FINAL round's predictions are needed (offline HD95/ASSD
        # surface backfill operates on the final record). Saving the heavy val-mask
        # npz every round was ~7x the NFS write traffic for no analysis use, and the
        # cluster NFS is the throughput bottleneck. Final-round-only keeps surface
        # capability; per-round masks are dropped (does not affect any metric/model).
        save_this_round = cfg.save_predictions and is_final
        metrics = eval_segmentation(
            model, val_subset, num_classes=num_classes,
            image_size=cfg.train.image_size, device=cfg.device,
            compute_surface=do_surface,
            save_preds=save_this_round, save_probs=cfg.save_logits and is_final,
        )
        t_eval = time.time() - t1
        # frozen_v3 prediction saving: dump the heavy preds to a compressed sidecar
        # and strip them from the metrics dict so the JSONL stays lean + serializable.
        predictions_path = ""
        _preds = metrics.pop("_preds", None)
        if _preds is not None:
            predictions_path = write_predictions(
                os.path.dirname(os.path.abspath(cfg.out_jsonl)),
                run_id, r, _preds, save_logits=cfg.save_logits)

        # 3. (If not last round) select next batch
        selected_ids: list[str] = []
        selected_scores: list[float] = []
        selected_pred_fg_ratio: list[float] = []
        selected_pred_class_dist: list[list[float]] = []
        selected_local_indices: list[int] = []
        diagnostics: dict = {}
        foundation_meta: dict = {}
        candidate_scores_path = ""
        candidate_count = 0
        feature_cache_keys: dict = {}
        sam_model_type = ""
        sam_checkpoint = ""
        t_select = 0.0
        if r + 1 < len(cfg.budget_plan):
            t2 = time.time()
            # Only compute what THIS policy actually needs (per its declared contract):
            # avoids redundant pred-cache / task-features / SAM extraction for methods
            # that don't use them (e.g. P0 needs nothing) — big speedup + no SAM OOM on
            # non-foundation methods.
            _need_feats = getattr(policy, "needs_features", ())
            valid_bboxes = _valid_bboxes_for(unlabeled_ds, cfg.train.image_size)
            # Build the per-round prediction inputs the policy needs. For streaming
            # policies (needs_pred_cache_probs=False, e.g. P1/P5/P6) we run the
            # batched forward ONCE and accumulate only the small per-sample (N,)
            # reductions + the small uint8 argmax — the full (N,C,H,W) probs is never
            # materialized (bounds RAM to O(batch*C*H*W)). Other policies (P9) get the
            # full probs cache as before.
            pred_cache = None
            streamed_reduce = None
            if getattr(policy, "needs_pred_cache", False):
                _img_indices = list(range(len(unlabeled_ds)))
                if not getattr(policy, "needs_pred_cache_probs", True):
                    reduced, argmax, fnames = stream_pool_reduce(
                        model,
                        _iter_images(unlabeled_ds, _img_indices, cfg.train.image_size),
                        per_batch_fn=policy.per_batch_reduce,
                        device=cfg.device,
                        valid_bboxes=valid_bboxes,
                    )
                    pred_cache = PredictionCache(probs=None, argmax=argmax, fnames=fnames)
                    streamed_reduce = reduced
                else:
                    pred_cache = build_prediction_cache(
                        model,
                        _iter_images(unlabeled_ds, _img_indices, cfg.train.image_size),
                        device=cfg.device,
                    )
            task_features = {
                "task_unet_pool":  extract_task_unet_features(model, unlabeled_ds, image_size=cfg.train.image_size, device=cfg.device),
                "task_unet_label": extract_task_unet_features(model, labeled_ds,   image_size=cfg.train.image_size, device=cfg.device),
            } if any("task_unet" in f for f in _need_feats) else {}
            if any("foundation" in f for f in _need_feats):
                foundation_features, foundation_meta = cfg.foundation_features_fn(
                    unlabeled_ds=unlabeled_ds, labeled_ds=labeled_ds,
                    seed=cfg.seed, device=cfg.device,
                )
            else:
                foundation_features, foundation_meta = {}, {}
            ctx = PolicyContext(
                seed=cfg.seed, round_idx=r, model=model, pred_cache=pred_cache,
                pool=unlabeled_ds, labeled=labeled_ds,
                features={**task_features, **foundation_features},
                num_classes=num_classes,
                valid_bboxes=valid_bboxes,
                query_seed=cseeds["query_seed"],
                streamed_reduce=streamed_reduce,
            )
            scores = policy.score(ctx)
            next_target = cfg.budget_plan[r + 1]
            k = next_target - len(labeled_local)
            k = max(0, min(k, len(unlabeled_ds)))
            selected_local_in_unlabeled = policy.select(ctx, scores, k=k)
            # firewall + budget assertions (constraint #3/#9): no duplicate
            # selection within a round, and no re-selection of an already
            # labeled sample (would corrupt budget counts / leak prior picks).
            assert len(set(selected_local_in_unlabeled)) == len(selected_local_in_unlabeled), \
                f"{cfg.policy_id}: duplicate indices in selection at round {r}"
            _picked_pool_local = {unlabeled_indices[i] for i in selected_local_in_unlabeled}
            assert _picked_pool_local.isdisjoint(labeled_local), \
                f"{cfg.policy_id}: selected an already-labeled sample at round {r} (leakage)"
            selected_ids = [unlabeled_ds[i].sample_id for i in selected_local_in_unlabeled]

            # per-selected enrichment from the prediction cache.
            # scores may be None for policies that don't override Policy.score()
            # (which defaults to None). At present: P0 Random, P3 CoreSet,
            # P4 BADGE, P7 SAM-CoreSet, P8 SAM-TypiClust. Policies that DO
            # return a per-image score: P1 Entropy, P2 BALD, P5 Entropy→CoreSet,
            # P6 PEAL, P9 PAAL.
            if scores is None:
                scores_arr = None
            elif hasattr(scores, "detach"):
                scores_arr = scores.detach().cpu().numpy()
            else:
                scores_arr = scores
            for i in selected_local_in_unlabeled:
                selected_scores.append(
                    float(scores_arr[i]) if scores_arr is not None else float("nan")
                )
                # predicted-class diagnostics need the prediction cache; methods that
                # don't build one (P0/P2/P3/P4/P7/P8) skip these signals.
                if pred_cache is None:
                    continue
                arg = pred_cache.argmax[i]                       # (H, W) int64
                total = int(arg.numel())
                selected_pred_fg_ratio.append(float((arg > 0).sum().item()) / max(1, total))
                per_c = [
                    float((arg == c).sum().item()) / max(1, total)
                    for c in range(num_classes)
                ]
                selected_pred_class_dist.append(per_c)

            for i in selected_local_in_unlabeled:
                pool_local = unlabeled_indices[i]
                labeled_local.add(pool_local)
            selected_local_indices = [unlabeled_indices[i] for i in selected_local_in_unlabeled]
            diagnostics = _sanitize_diagnostics(ctx.diagnostics_out)

            # candidate scores for EVERY unlabeled candidate this round -> sidecar
            candidate_ids = [unlabeled_ds[i].sample_id for i in range(len(unlabeled_ds))]
            if scores_arr is not None:
                candidate_scores_list = [float(x) for x in np.asarray(scores_arr).reshape(-1)]
            else:
                candidate_scores_list = [float("nan")] * len(candidate_ids)
            out_dir = os.path.dirname(os.path.abspath(cfg.out_jsonl))
            candidate_scores_path = write_candidate_scores(
                out_dir, run_id, r, candidate_ids, candidate_scores_list)
            candidate_count = len(candidate_ids)

            # feature-cache keys for feature-based methods (P3/P4/P7/P8)
            if cfg.policy_id in ("P3", "P5", "P9"):
                feature_cache_keys["task_unet"] = f"task_unet_bottleneck@{ckpt_hash[:12]}"
            if cfg.policy_id in ("P4", "P4b"):
                feature_cache_keys["badge_grad"] = f"ce_head_grad@{ckpt_hash[:12]}"
            if cfg.policy_id in ("P7", "P8", "P8b") and foundation_meta:
                feature_cache_keys["foundation"] = (
                    f"{foundation_meta.get('encoder_id','')}__{foundation_meta.get('cache_version','')}")
                sam_model_type = str(foundation_meta.get("model_type", ""))
                sam_checkpoint = str(foundation_meta.get("checkpoint", ""))
            t_select = time.time() - t2

        rec = TrajectoryRecord(
            run_id=run_id, round=r,
            dataset=adapter.name, modality=adapter.modality, target=adapter.target,
            dim=adapter.dim, query_unit=adapter.query_unit, seed=cfg.seed,
            round_seed=round_seed, component_seeds=cseeds,
            total_rounds=_total_rounds, budget_denominator=_budget_denominator,
            predictions_path=predictions_path,
            policy_id=cfg.policy_id, policy_name=policy_class_name,
            method_version=policy.version, is_ablation=policy.is_ablation,
            policy_config=cfg.policy_config,
            cumulative_budget=cum_target,
            incremental_query_count=len(selected_ids),
            labeled_count=len(labeled_local) - len(selected_local_indices),
            labeled_ratio=(len(labeled_local) - len(selected_local_indices)) / max(1, len(pool_subset)),
            selected_ids=selected_ids,
            selected_scores=selected_scores,
            selected_pred_fg_ratio=selected_pred_fg_ratio,
            selected_pred_class_dist=selected_pred_class_dist,
            candidate_count=candidate_count,
            candidate_scores_path=candidate_scores_path,
            initial_labeled_path=init_path,
            initial_labeled_ids=(list(init_ids) if r == 0 else []),
            selection_diagnostics=diagnostics,
            training={
                "num_iters": cfg.train.num_iters,
                "batch_size": cfg.train.batch_size,
                "lr": cfg.train.lr,
                "image_size": cfg.train.image_size,
                "aspect_preserve": cfg.train.aspect_preserve,
                "resolution_policy": ("longside_pad32" if cfg.train.aspect_preserve else "square"),
                "preproc_cache": _cache_status,
                "train_runtime_sec": t_train,
                "query_runtime_sec": t_select,
                **train_stats,
            },
            ckpt_id=f"{run_id}__r{r}",
            ckpt_path=ckpt_path,
            ckpt_hash=ckpt_hash,
            config_hash=cfg_hash,
            feature_cache_keys=feature_cache_keys,
            sam_model_type=sam_model_type,
            sam_checkpoint=sam_checkpoint,
            metrics=metrics,
            runtime_sec={"train": t_train, "eval": t_eval, "select": t_select},
            gpu_mem_mb=_peak_gpu_mb() if _cuda else None,
            gpu_name=_gpu_name, gpu_total_mem_mb=_gpu_total_mb,
            foundation=foundation_meta if cfg.policy_id in ("P7", "P8", "P8b") else {},
        )
        append_record(cfg.out_jsonl, rec)
        records.append({"round": r, "metrics": metrics, "train": train_stats, "selected_n": len(selected_ids)})

    return records
