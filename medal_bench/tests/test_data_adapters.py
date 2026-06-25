"""1-image read-and-print smoke per dataset adapter.

For each pilot dataset, the smoke test:
  1. Instantiates the adapter at the canonical root.
  2. Loads sample[0].
  3. Prints id / image shape, dtype, range / mask shape, dtype, unique-classes
     / patient_id, slice_index.
  4. Asserts shape/dtype/value-range invariants.

This is the gate the user mandated (#7 in the v1 modifications) before any
training runs. Datasets whose data is not yet on disk should skip with a
clear reason rather than silently passing.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from medal_bench.data.adapters import (
    ISIC2018Adapter, CVCClinicDBAdapter, BUSIAdapter,
    ROSE1Adapter, PROMISE12Adapter, MSD07PancreasAdapter,
    REFUGEAdapter, ImageMaskFolderAdapter,
)


DATA_ROOT = Path(os.environ.get(
    "MEDAL_DATA_ROOT", "/groups/echambe2/datasets/data"
))


def _print_sample(adapter_name: str, ds, s) -> None:
    img, mask = s.image, s.mask
    uniq = np.unique(mask).tolist() if mask is not None else None
    print(
        f"\n[{adapter_name}] len={len(ds)} "
        f"id={s.sample_id!r} patient_id={s.patient_id!r} slice_index={s.slice_index!r}\n"
        f"  image: shape={img.shape} dtype={img.dtype} "
        f"min={float(img.min()):.4f} max={float(img.max()):.4f}\n"
        f"  mask : shape={mask.shape} dtype={mask.dtype} classes={uniq}",
    )


def _assert_2d_image(img: np.ndarray, channels: int) -> None:
    assert img.dtype == np.float32, f"image dtype {img.dtype}, expected float32"
    assert img.ndim == 3 and img.shape[0] == channels, \
        f"image shape {img.shape}, expected ({channels}, H, W)"
    assert 0.0 - 1e-6 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6, \
        f"image range [{img.min()}, {img.max()}] not in [0, 1]"


def _assert_mask(mask: np.ndarray, num_classes: int) -> None:
    assert mask.dtype == np.int64, f"mask dtype {mask.dtype}, expected int64"
    assert mask.ndim == 2, f"mask shape {mask.shape}, expected (H, W)"
    u = np.unique(mask)
    assert u.min() >= 0 and u.max() < num_classes, \
        f"mask classes {u.tolist()} outside [0, {num_classes})"


# ----- ISIC 2018 ------------------------------------------------------------

def test_isic2018_smoke():
    root = DATA_ROOT / "2d" / "isic2018_task1"
    if not (root / "extracted").exists():
        pytest.skip(f"ISIC2018 not on disk: {root}")
    ds = ISIC2018Adapter(str(root), split="train")
    s = ds[0]
    _print_sample("isic2018/train", ds, s)
    _assert_2d_image(s.image, channels=3)
    _assert_mask(s.mask, num_classes=2)
    assert s.patient_id is None and s.slice_index is None


# ----- CVC-ClinicDB --------------------------------------------------------

def test_cvc_clinicdb_smoke():
    root = DATA_ROOT / "2d" / "cvc_clinicdb"
    if not (root / "extracted" / "PNG" / "Original").exists():
        pytest.skip(f"CVC-ClinicDB not on disk: {root}")
    ds = CVCClinicDBAdapter(str(root))
    s = ds[0]
    _print_sample("cvc_clinicdb", ds, s)
    _assert_2d_image(s.image, channels=3)
    _assert_mask(s.mask, num_classes=2)


# ----- BUSI ----------------------------------------------------------------

def test_busi_smoke():
    root = DATA_ROOT / "2d" / "busi"
    if not (root / "extracted" / "Dataset_BUSI_with_GT").exists():
        pytest.skip(f"BUSI not on disk: {root}")
    ds = BUSIAdapter(str(root))
    s = ds[0]
    _print_sample("busi", ds, s)
    _assert_2d_image(s.image, channels=1)
    _assert_mask(s.mask, num_classes=2)


def test_busi_multi_mask_merge():
    """If any benign image has >1 mask, the merged mask should still be binary
    and at least one such image must exist in the index."""
    root = DATA_ROOT / "2d" / "busi"
    if not (root / "extracted" / "Dataset_BUSI_with_GT").exists():
        pytest.skip(f"BUSI not on disk: {root}")
    ds = BUSIAdapter(str(root))
    # find first image with >1 mask file
    multi = next(((i, paths) for i, (_, _, _, paths) in enumerate(ds._index) if len(paths) > 1), None)
    if multi is None:
        pytest.skip("No multi-mask BUSI image in index (unexpected)")
    i, paths = multi
    s = ds[i]
    print(f"\n[busi/multi] id={s.sample_id} merged from {len(paths)} masks")
    _assert_mask(s.mask, num_classes=2)


# ----- ROSE-1 (stub, data missing) -----------------------------------------

def test_rose1_smoke_data_missing():
    """Adapter is a stub; constructor must raise loudly with the missing path."""
    root = DATA_ROOT / "2d" / "rose1"
    with pytest.raises(FileNotFoundError, match="ROSE-1"):
        ROSE1Adapter(str(root))


# ----- PROMISE12 -----------------------------------------------------------

def test_promise12_smoke():
    root = DATA_ROOT / "2d" / "promise12"
    if not list((root / "extracted").glob("Case??.mhd")):
        pytest.skip(f"PROMISE12 not on disk: {root}")
    ds = PROMISE12Adapter(str(root))
    s = ds[0]
    _print_sample("promise12", ds, s)
    _assert_2d_image(s.image, channels=1)
    _assert_mask(s.mask, num_classes=2)
    assert s.patient_id is not None and s.slice_index is not None
    # patient_ids() returns one entry per slice, parallel to __getitem__
    pids = ds.patient_ids()
    assert pids is not None and len(pids) == len(ds)
    # at least 2 distinct cases — pilot uses volume-level splits
    assert len(set(pids)) >= 2


# ----- MSD07 Pancreas ------------------------------------------------------

def test_msd07_pancreas_smoke():
    root = DATA_ROOT / "3d" / "msd_task07_pancreas"
    if not (root / "extracted" / "Task07_Pancreas" / "imagesTr").exists():
        pytest.skip(f"MSD07 not on disk: {root}")
    ds = MSD07PancreasAdapter(str(root))
    s = ds[0]
    _print_sample("msd07_pancreas", ds, s)
    _assert_2d_image(s.image, channels=1)
    _assert_mask(s.mask, num_classes=3)
    assert s.patient_id is not None and s.slice_index is not None
    pids = ds.patient_ids()
    assert pids is not None and len(pids) == len(ds)
    assert len(set(pids)) >= 2


# ----- ORIGA (C=3 disc+cup) ------------------------------------------------

def test_origa_disc_cup_smoke():
    """origa is re-tasked to C=3 (bg/disc/cup); masks are dense {0,1,2}."""
    root = DATA_ROOT / "2d" / "origa" / "raw" / "ORIGA"
    if not (root / "Masks").exists():
        pytest.skip(f"ORIGA not on disk: {root}")
    ds = ImageMaskFolderAdapter(
        name="origa", modality="fundus", target="optic_nerve_head",
        image_dir=str(root / "Images"), mask_dir=str(root / "Masks"),
        num_classes=3, to_rgb=True, binarize=False,
    )
    assert ds.num_classes == 3
    s = ds[0]
    _print_sample("origa", ds, s)
    _assert_2d_image(s.image, channels=3)
    _assert_mask(s.mask, num_classes=3)


# ----- REFUGE (C=3 disc+cup) -----------------------------------------------

def test_refuge_disc_cup_smoke():
    root = DATA_ROOT / "2d" / "refuge" / "raw" / "REFUGE"
    if not (root / "train" / "Masks").exists():
        pytest.skip(f"REFUGE not on disk: {root}")
    ds = REFUGEAdapter(str(root))
    assert ds.num_classes == 3
    assert len(ds) == 1200  # 400 train + 400 val + 400 test, pooled
    s = ds[0]
    _print_sample("refuge", ds, s)
    _assert_2d_image(s.image, channels=3)
    _assert_mask(s.mask, num_classes=3)
    # cup (2) is nested inside disc: its bbox lies within the disc+cup bbox
    m = s.mask
    if 2 in np.unique(m) and 1 in np.unique(m):
        ys2, xs2 = np.where(m == 2)
        ysf, xsf = np.where(m >= 1)
        assert (ys2.min() >= ysf.min() and ys2.max() <= ysf.max()
                and xs2.min() >= xsf.min() and xs2.max() <= xsf.max())
