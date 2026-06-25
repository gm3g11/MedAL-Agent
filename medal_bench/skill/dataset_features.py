"""Static dataset descriptors (Block A) for the 19-set -- computed ONCE from the
data adapters and cached. These are decision-time features available before any
AL session: modality, object family, dim, class count, pool size, foreground
sparsity / class imbalance, image geometry. No trajectory / no test metric here.

Run:  python -m medal_bench.skill.dataset_features
Writes: runs/frozen_v5/skill/dataset_features.csv
"""
from __future__ import annotations

import csv
import os

import numpy as np

from medal_bench.data.adapters import DATASET_REGISTRY
from medal_bench.skill.schema import DS19, SKILL_DIR

DATA_ROOT = "/groups/echambe2/datasets/data"
N_FG_SAMPLES = 60   # evenly-spaced images sampled per dataset for FG statistics

# Coarse anatomical object family (from DATASET_TABLE_FINAL.md). Kept explicit and
# auditable rather than parsed, so the grouping is transparent.
OBJECT_FAMILY = {
    "btcv_synapse": "abdomen_multi", "flare22": "abdomen_multi",
    "ext_abdoment1k": "abdomen_multi", "msd_task07_pancreas": "pancreas",
    "msd_task03_liver": "liver", "liqa_mri": "liver", "kits19": "kidney",
    "msd_task09_spleen": "spleen", "kvasir_seg": "gi_polyp",
    "mmwhs_ct": "cardiac", "hvsmr2016": "cardiac", "care_leftatrium_2026": "cardiac",
    "ext_brats2020": "brain", "msd_task04_hippocampus": "brain",
    "refuge": "eye", "origa": "eye", "isic2018": "skin",
    "glas2015": "histo_gland", "busi": "breast",
}


def _fg_stats(ds, n_classes: int):
    """Mean/median FG fraction, rarest-class fraction, class imbalance over a
    deterministic even sample of images."""
    n = len(ds)
    idx = np.linspace(0, n - 1, min(N_FG_SAMPLES, n)).round().astype(int)
    idx = sorted(set(idx.tolist()))
    fg_fracs = []
    class_fracs = np.zeros(max(n_classes, 2), dtype=np.float64)
    cnt = 0
    h = w = 0
    for i in idx:
        s = ds[i]
        m = np.asarray(s.mask)
        img = np.asarray(s.image)
        hw = img.shape[-2:]
        h, w = int(hw[0]), int(hw[1])
        fg_fracs.append(float((m > 0).mean()))
        for c in range(len(class_fracs)):
            class_fracs[c] += float((m == c).mean())
        cnt += 1
    class_fracs /= max(cnt, 1)
    fg = np.array(fg_fracs)
    # rarest foreground class = smallest mean fraction among classes 1..C-1
    fg_class = class_fracs[1:n_classes] if n_classes > 1 else class_fracs[1:]
    rarest = float(fg_class[fg_class > 0].min()) if (fg_class > 0).any() else float(fg.mean())
    bg = float(class_fracs[0]) if class_fracs[0] > 0 else 1.0
    mean_fg = float(fg.mean()) if fg.mean() > 0 else 1e-6
    imbalance = bg / mean_fg
    return dict(
        fg_frac_mean=round(float(fg.mean()), 6),
        fg_frac_median=round(float(np.median(fg)), 6),
        rarest_class_frac=round(rarest, 6),
        class_imbalance=round(float(imbalance), 4),
        img_h=h, img_w=w,
        aspect_ratio=round(h / w, 4) if w else 1.0,
    )


def compute(out_path: str | None = None) -> list[dict]:
    rows = []
    for d in DS19:
        ds = DATASET_REGISTRY[d](DATA_ROOT)
        n = len(ds)
        s0 = ds[0]
        n_classes = int(getattr(ds, "num_classes", 0) or (np.asarray(s0.mask).max() + 1))
        modality = getattr(ds, "modality", "?")
        target = getattr(ds, "target", "?")
        pids = ds.patient_ids() if hasattr(ds, "patient_ids") else None
        n_groups = len(set(pids)) if pids else n
        stats = _fg_stats(ds, n_classes)
        rows.append(dict(
            dataset=d, modality=modality, target=target,
            object_family=OBJECT_FAMILY[d], n_classes=n_classes,
            is_multiclass=int(n_classes > 2), n_images=n, n_groups=n_groups,
            slices_per_case=round(n / max(n_groups, 1), 3), **stats,
        ))
        print(f"  {d:24s} mod={modality:11s} C={n_classes:2d} n_img={n:6d} "
              f"groups={n_groups:4d} fg={stats['fg_frac_mean']:.4f} "
              f"rarest={stats['rarest_class_frac']:.4f}", flush=True)
    if out_path is None:
        out_path = os.path.join(SKILL_DIR, "dataset_features.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} datasets)")
    return rows


if __name__ == "__main__":
    compute()
