"""Per-round JSONL trajectory logging.

Schema is versioned (TRAJECTORY_SCHEMA_VERSION). Any change is a new version;
the analysis layer keeps a compatibility table.

One JSONL file per ``run_id``; one record per AL round (including round 0).
Records are appended atomically (open, write+newline+flush+close per record),
so a partial run leaves a valid prefix.
"""
from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Any

TRAJECTORY_SCHEMA_VERSION = "v3"


@dataclass
class TrajectoryRecord:
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    run_id: str = ""
    round: int = 0

    # run/dataset identity (constant across rounds within a run)
    dataset: str = ""
    modality: str = ""
    target: str = ""
    dim: str = ""                  # "2d" | "3d"
    query_unit: str = ""           # "image" | "slice" | "volume"
    seed: int = 0

    # frozen_v3 component seeding: round_seed = cfg.seed + r; component_seeds holds
    # {model_init_seed, loader_seed, query_seed, dropout_seed} derived from it. Logged
    # every round so each round's reproducibility is independently auditable.
    round_seed: int = 0
    component_seeds: dict = field(default_factory=dict)

    # total number of AL rounds for this cell (== len(budget_plan)); lets the
    # dispatcher decide "done" by round count, not file existence (lease-race safe).
    total_rounds: int = 0

    # budget denominator provenance (frozen_v3 / M2): full_train_N, requested_pool_cap,
    # actual_AL_pool_N, fraction_of_AL_pool, fraction_of_full_train, abs budget_plan.
    budget_denominator: dict = field(default_factory=dict)

    # policy identity (constant)
    policy_id: str = ""
    policy_name: str = ""
    method_version: str = ""        # policy.version
    is_ablation: bool = False       # True for P4b/P8b
    policy_config: dict = field(default_factory=dict)

    # budget bookkeeping
    cumulative_budget: int = 0      # target #labeled at this round (budget_plan[r])
    incremental_query_count: int = 0  # #samples queried this round (k)

    # round-level state (changes each round)
    labeled_count: int = 0
    labeled_ratio: float = 0.0
    selected_ids: list = field(default_factory=list)
    selected_scores: list = field(default_factory=list)
    selected_pred_fg_ratio: list = field(default_factory=list)
    selected_pred_class_dist: list = field(default_factory=list)

    # candidate scores for EVERY unlabeled sample this round (saved to a sidecar
    # to keep the JSONL lean); path + count recorded here.
    candidate_count: int = 0
    candidate_scores_path: str = ""

    # compressed val-prediction dump for this round (masks + ids + valid masks +
    # optional fp16 probs); "" when prediction saving is off. See write_predictions.
    predictions_path: str = ""

    # initial labeled set (the cold start, shared across methods)
    initial_labeled_path: str = ""
    initial_labeled_ids: list = field(default_factory=list)  # populated on round 0 only

    # selection diagnostics (P3 fallback counts, batch diversity, etc.)
    selection_diagnostics: dict = field(default_factory=dict)

    # training metadata + checkpoint provenance for the query-time model
    training: dict = field(default_factory=dict)
    ckpt_id: str = ""
    ckpt_path: str = ""             # set when checkpoints are saved to disk
    ckpt_hash: str = ""             # sha256 of the model state_dict at query time
    config_hash: str = ""           # hash of the run/preprocessing config

    # feature-cache keys for feature-based methods (P3/P4/P7/P8)
    feature_cache_keys: dict = field(default_factory=dict)
    # SAM provenance for P7/P8
    sam_model_type: str = ""
    sam_checkpoint: str = ""

    # evaluation metrics
    metrics: dict = field(default_factory=dict)

    # timings
    runtime_sec: dict = field(default_factory=dict)
    # peak CUDA memory (MB) for the round; None on CPU
    gpu_mem_mb: float | None = None
    # GPU identity for the cell (constant across rounds); "" / None on CPU
    gpu_name: str = ""
    gpu_total_mem_mb: float | None = None

    # foundation-encoder identity for P7/P8 (or ""/None for other policies)
    foundation: dict = field(default_factory=dict)

    notes: str = ""


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def state_dict_hash(model) -> str:
    """sha256 of the model state_dict (sorted keys), so the exact query-time
    weights are reproducibly identified."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_candidate_scores(out_dir: str, run_id: str, rnd: int,
                           candidate_ids: list, candidate_scores: list) -> str:
    """Write per-candidate scores for one round to a sidecar JSON; return path."""
    cdir = osp.join(out_dir, "candidate_scores")
    os.makedirs(cdir, exist_ok=True)
    path = osp.join(cdir, f"{run_id}__r{rnd}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"run_id": run_id, "round": rnd,
                   "candidate_ids": list(candidate_ids),
                   "candidate_scores": list(candidate_scores)}, f)
    os.replace(tmp, path)
    return path


def write_predictions(out_dir: str, run_id: str, rnd: int, preds: dict,
                      save_logits: bool = False) -> str:
    """Dump one round's val predictions to a compressed ``.npz``; return its path.

    ``preds`` carries (all aligned to val order):
      sample_ids (str[N]), pred (uint8 N,H,W), gt (uint8 N,H,W),
      patient_ids (str[N], "" if none), slice_indices (int[N], -1 if none),
      valid_bbox (int[N,4] = y0,x0,h,w), and optionally probs (fp16 N,C,H,W).
    Spacing/affine are not available today (3D loaders discard headers) -> stored
    as a 'unavailable' slot + units='pixels' so native-mm metrics can be backfilled
    later from the same dump without retraining. Masks/ids/valid are ALWAYS saved;
    probs only when save_logits. Atomic (tmp + os.replace)."""
    import numpy as _np
    pdir = osp.join(out_dir, "predictions")
    os.makedirs(pdir, exist_ok=True)
    path = osp.join(pdir, f"{run_id}__r{rnd}.npz")
    tmp = path + ".tmp.npz"
    arrays = {
        "sample_ids": _np.asarray(preds["sample_ids"]),
        "pred": _np.asarray(preds["pred"], dtype=_np.uint8),
        "gt": _np.asarray(preds["gt"], dtype=_np.uint8),
        "patient_ids": _np.asarray(preds["patient_ids"]),
        "slice_indices": _np.asarray(preds["slice_indices"], dtype=_np.int32),
        "valid_bbox": _np.asarray(preds["valid_bbox"], dtype=_np.int32),
        "spacing": _np.array("unavailable"),
        "units": _np.array("pixels"),
    }
    if save_logits and preds.get("probs") is not None:
        arrays["probs"] = _np.asarray(preds["probs"], dtype=_np.float16)
    _np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def append_record(path: str, record: TrajectoryRecord | dict) -> None:
    """Append one record as a JSON line. Creates parent dirs as needed."""
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    if hasattr(record, "__dataclass_fields__"):
        # dataclass -> dict
        from dataclasses import asdict
        d = asdict(record)
    else:
        d = dict(record)
    with open(path, "a") as f:
        f.write(json.dumps(d, sort_keys=False) + "\n")
        f.flush()


def read_jsonl(path: str) -> list[dict]:
    """Read all records from a JSONL file. Returns [] if file does not exist."""
    if not osp.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_trajectory_deduped(path: str) -> list[dict]:
    """Read a trajectory, collapsing DUPLICATE ROUNDS and de-duplicating
    ``selected_ids`` — for analysis / skill export.

    Some early Stage-1 cells were double-run during worker churn (pre atomic-rename
    fix), leaving the same round index appended twice. Keep the LAST occurrence per
    round index (identical content; last is the completed write), and drop any
    repeated ids inside ``selected_ids`` (order-preserving). Non-destructive: does
    not modify the file. ``metrics``/results are unaffected (duplicate rounds are
    identical), this only gives a clean per-round selection set for export.
    """
    by_round: dict[int, dict] = {}
    for rec in read_jsonl(path):
        by_round[rec.get("round", 0)] = rec        # last wins
    out = []
    for rnd in sorted(by_round):
        rec = by_round[rnd]
        sids = rec.get("selected_ids")
        if isinstance(sids, list):
            rec = dict(rec)
            rec["selected_ids"] = list(dict.fromkeys(sids))   # order-preserving unique
        out.append(rec)
    return out
