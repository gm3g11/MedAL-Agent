"""Per-dataset adapters + a central DATASET_REGISTRY.

DATASET_REGISTRY maps a short dataset name -> factory(data_root) -> MedALDataset.
Both ``runner.run_one`` and ``runner.smoke_matrix`` resolve datasets through it,
so adding a dataset is a single registry line (use ImageMaskFolderAdapter for
the common image+mask-folder layout, or a bespoke adapter otherwise).
"""
from medal_bench.data.adapters.isic2018 import ISIC2018Adapter
from medal_bench.data.adapters.cvc_clinicdb import CVCClinicDBAdapter
from medal_bench.data.adapters.busi import BUSIAdapter
from medal_bench.data.adapters.rose1 import ROSE1Adapter
from medal_bench.data.adapters.promise12 import PROMISE12Adapter
from medal_bench.data.adapters.msd07_pancreas import MSD07PancreasAdapter
from medal_bench.data.adapters.mmwhs import MMWHSAdapter
from medal_bench.data.adapters.refuge import REFUGEAdapter
from medal_bench.data.adapters.generic import ImageMaskFolderAdapter
# expansion adapters (TIER-2/3/4, written + CPU-tested + leakage-verified 2026-06-23)
from medal_bench.data.adapters.nlm_montgomery import NLMMontgomeryAdapter
from medal_bench.data.adapters.tnbc import TNBCAdapter
from medal_bench.data.adapters.drive import DRIVEAdapter
from medal_bench.data.adapters.pannuke import PanNukeAdapter
from medal_bench.data.adapters.cholecseg8k import CholecSeg8kAdapter
from medal_bench.data.adapters.acdc import ACDCAdapter
from medal_bench.data.adapters.duke_dme_chiu2015 import DukeDMEChiu2015Adapter
from medal_bench.data.adapters.umn_oct import UMNOCTAdapter
# object-filler adapters (downloaded 2026-06-23: vessels, EM, prostate, dermoscopy, breast-US)
from medal_bench.data.adapters.chase_db1 import ChaseDB1Adapter
from medal_bench.data.adapters.isbi2012_em import ISBI2012EMAdapter
from medal_bench.data.adapters.msd_task05_prostate import MSDTask05ProstateAdapter
from medal_bench.data.adapters.ph2 import PH2Adapter
from medal_bench.data.adapters.bus_bra import BUSBRAAdapter
from medal_bench.data.adapters.crag import CRAGAdapter
from medal_bench.data.adapters.fives import FIVESAdapter
from medal_bench.data.adapters.cremi import CREMIAdapter
from medal_bench.data.adapters.snemi3d import SNEMI3DAdapter

__all__ = [
    "ISIC2018Adapter", "CVCClinicDBAdapter", "BUSIAdapter",
    "ROSE1Adapter", "PROMISE12Adapter", "MSD07PancreasAdapter",
    "MMWHSAdapter", "REFUGEAdapter", "ImageMaskFolderAdapter", "DATASET_REGISTRY",
]


# ----- generic image+mask-folder datasets (verified by sample-load 2026-06-12) -----

def _kvasir_seg(dr):
    r = f"{dr}/2d/kvasir_seg/extracted/Kvasir-SEG"
    return ImageMaskFolderAdapter(name="kvasir_seg", modality="endoscopy", target="polyp",
        image_dir=f"{r}/images", mask_dir=f"{r}/masks",
        num_classes=2, to_rgb=True, binarize=True, bin_threshold=128)


def _hyperkvasir_seg(dr):
    r = f"{dr}/2d/hyperkvasir_seg/extracted/segmented-images"
    return ImageMaskFolderAdapter(name="hyperkvasir_seg", modality="endoscopy", target="polyp",
        image_dir=f"{r}/images", mask_dir=f"{r}/masks",
        num_classes=2, to_rgb=True, binarize=True, bin_threshold=128)


def _glas2015(dr):
    r = f"{dr}/2d/glas2015/extracted/Warwick QU Dataset (Released 2016_07_08)"
    return ImageMaskFolderAdapter(name="glas2015", modality="histology", target="gland",
        image_dir=r, mask_suffix="_anno",
        num_classes=2, to_rgb=True, binarize=True, bin_threshold=1)


def _origa(dr):
    # C=3 disc+cup (origa_disc_cup): Masks/ PNGs are already dense {0=bg,1=disc,2=cup}
    # (cup nested inside disc); read as raw class indices, NOT binarized.
    r = f"{dr}/2d/origa/raw/ORIGA"
    return ImageMaskFolderAdapter(name="origa", modality="fundus", target="optic_nerve_head",
        image_dir=f"{r}/Images", mask_dir=f"{r}/Masks",
        num_classes=3, to_rgb=True, binarize=False)


def _g1020(dr):
    # C=3 disc+cup, identical encoding to origa (Masks/ dense {0,1,2}); .json sidecars in
    # Images/ are auto-skipped by the adapter's image-extension filter. De-singletons fundus.
    r = f"{dr}/2d/g1020/raw/G1020"
    return ImageMaskFolderAdapter(name="g1020", modality="fundus", target="optic_nerve_head",
        image_dir=f"{r}/Images", mask_dir=f"{r}/Masks",
        num_classes=3, to_rgb=True, binarize=False)


def _jsrt_scr(dr):
    # C=2 binary lung-field on grayscale chest X-ray (masks {0,1}); first CXR modality.
    # images nested at extracted/images/images; JPCLN001.png <-> masks/JPCLN001.tif by stem.
    r = f"{dr}/2d/jsrt_scr/extracted"
    return ImageMaskFolderAdapter(name="jsrt_scr", modality="chest_xray", target="lung_field",
        image_dir=f"{r}/images/images", mask_dir=f"{r}/masks",
        num_classes=2, to_rgb=False, binarize=True, bin_threshold=1)


DATASET_REGISTRY = {
    # bespoke adapters (original pilot set)
    "isic2018":       lambda dr: ISIC2018Adapter(f"{dr}/2d/isic2018_task1", split="train"),
    "cvc_clinicdb":   lambda dr: CVCClinicDBAdapter(f"{dr}/2d/cvc_clinicdb"),
    "busi":           lambda dr: BUSIAdapter(f"{dr}/2d/busi"),
    "promise12":      lambda dr: PROMISE12Adapter(f"{dr}/2d/promise12"),
    "msd07_pancreas": lambda dr: MSD07PancreasAdapter(f"{dr}/3d/msd_task07_pancreas"),
    # multiclass 3D->2D (remap-validated 2026-06-13); CT & MR kept separate
    "mmwhs_ct":       lambda dr: MMWHSAdapter(f"{dr}/3d/mmwhs", modality="ct"),
    "mmwhs_mr":       lambda dr: MMWHSAdapter(f"{dr}/3d/mmwhs", modality="mr"),
    # generic image+mask-folder datasets
    "kvasir_seg":      _kvasir_seg,
    "hyperkvasir_seg": _hyperkvasir_seg,
    "glas2015":        _glas2015,
    "origa":           _origa,
    "g1020":           _g1020,
    "jsrt_scr":        _jsrt_scr,
    "refuge":          lambda dr: REFUGEAdapter(f"{dr}/2d/refuge/raw/REFUGE"),
    # expansion bespoke adapters (TIER-2/3/4)
    "nlm_montgomery":     lambda dr: NLMMontgomeryAdapter(f"{dr}/2d/nlm_montgomery"),
    "tnbc":               lambda dr: TNBCAdapter(f"{dr}/2d/tnbc"),
    "drive":              lambda dr: DRIVEAdapter(f"{dr}/2d/drive"),
    "pannuke":            lambda dr: PanNukeAdapter(f"{dr}/2d/pannuke"),
    "cholecseg8k":        lambda dr: CholecSeg8kAdapter(f"{dr}/2d/cholecseg8k"),
    "acdc":               lambda dr: ACDCAdapter(f"{dr}/2d/acdc"),
    "duke_dme_chiu2015":  lambda dr: DukeDMEChiu2015Adapter(f"{dr}/2d/duke_dme_chiu2015"),
    "umn_oct":            lambda dr: UMNOCTAdapter(f"{dr}/2d/umn_oct"),
    # object-filler downloads (vessels, EM, prostate, dermoscopy, breast-US)
    "chase_db1":           lambda dr: ChaseDB1Adapter(f"{dr}/2d/chase_db1"),
    "isbi2012_em":         lambda dr: ISBI2012EMAdapter(f"{dr}/2d/isbi2012_em"),
    "msd_task05_prostate": lambda dr: MSDTask05ProstateAdapter(f"{dr}/2d/msd_task05_prostate"),
    "ph2":                 lambda dr: PH2Adapter(f"{dr}/2d/ph2"),
    "bus_bra":             lambda dr: BUSBRAAdapter(f"{dr}/2d/bus_bra"),
    "crag":                lambda dr: CRAGAdapter(f"{dr}/2d/crag"),
    "fives":               lambda dr: FIVESAdapter(f"{dr}/2d/fives"),
    "cremi":               lambda dr: CREMIAdapter(f"{dr}/3d/cremi"),
    "snemi3d":             lambda dr: SNEMI3DAdapter(f"{dr}/3d/snemi3d"),
}

# ----- bridged: the 21 audited medal_agent datasets (single source of truth for
# the expanded 3D-sliced + multiclass set). No-op if medal_agent is unimportable. -----
from medal_bench.data.adapters.medal_agent_bridge import register_medal_agent_datasets  # noqa: E402

BRIDGED_DATASETS = register_medal_agent_datasets(DATASET_REGISTRY)
