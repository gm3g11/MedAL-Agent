"""SAM image-encoder features for P7 (SAM-CoreSet) and P8 (SAM-TypiClust).

Supports SAM ViT-B / ViT-L / ViT-H via ``sam_model_type``:
  - vit_b : HuggingFace ``facebook/sam-vit-base`` (default; backward-compatible
            cache key, no checkpoint download needed beyond the HF hub).
  - vit_h : original facebookresearch/segment-anything checkpoint
            (``sam_vit_h_4b8939.pth``) loaded via the ``segment_anything`` pkg.
  - vit_l : original checkpoint, same path (supply ``checkpoint=``).

All variants:
  Preprocess : replicate grayscale -> 3ch; resize to 1024x1024 bilinear;
               normalize with SAM pixel_mean / pixel_std (0-255 scale).
  Encoder    : image encoder -> (B, 256, 64, 64) for EVERY ViT size (the SAM
               neck always emits 256 channels), so FEATURE_DIM is 256 for all.
  Pool       : global average pool over the 64x64 grid -> (B, 256).

The pooled vector is the per-image feature P7/P8 cluster on.

HDF5 cache layout (one file per (dataset, model_type)):
  {cache_dir}/{dataset_name}__{encoder_id}__{preprocess_hash[:10]}.h5
The ``encoder_id`` embeds the model type (``segment_anything/vit_h/image_encoder``
vs ``facebook/sam-vit-base/vision_encoder``), so SAM-B and SAM-H features can
NEVER share a cache file. On open we verify the stored attrs match the current
spec; on mismatch we recompute.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from medal_bench.data.base import MedALDataset


# ----- preprocess config + hash --------------------------------------------

@dataclass(frozen=True)
class SamPreprocessConfig:
    target_size: int = 1024
    resize_mode: str = "bilinear"
    grayscale_to_rgb: str = "replicate"
    pixel_mean: tuple = (123.675, 116.28, 103.53)
    pixel_std: tuple = (58.395, 57.12, 57.375)
    input_scale_max: float = 255.0   # we scale our [0,1] inputs by 255 before normalize

    def hash(self) -> str:
        s = (
            f"target_size={self.target_size}|resize_mode={self.resize_mode}|"
            f"gray={self.grayscale_to_rgb}|mean={self.pixel_mean}|std={self.pixel_std}|"
            f"scale={self.input_scale_max}"
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----- encoder spec (resolves model_type -> backend + cache identity) -------

CACHE_VERSION = "v1"
FEATURE_DIM = 256                     # SAM neck output channels (all ViT sizes)
SAM_POOLING_RULE = "global_avg_pool_over_64x64"

# Default original-format checkpoints (override via ``checkpoint=`` / CLI).
_DEFAULT_ORIGINAL_CKPT = {
    "vit_h": "/groups/echambe2/gmeng/MedAL-Agent/sam_vit_h_4b8939.pth",
}


@dataclass(frozen=True)
class SamEncoderSpec:
    model_type: str               # vit_b | vit_l | vit_h
    backend: str                  # "hf" | "original"
    hf_id: Optional[str]
    checkpoint: Optional[str]
    encoder_id: str               # unique per model type; goes into the cache key
    feature_dim: int = FEATURE_DIM
    cache_version: str = CACHE_VERSION


def resolve_sam_spec(model_type: str = "vit_b",
                     checkpoint: Optional[str] = None) -> SamEncoderSpec:
    """Pick the backend + cache identity for a SAM model type.

    vit_b (no checkpoint) -> HuggingFace (backward-compatible cache key).
    Otherwise -> original ``segment_anything`` checkpoint (.pth)."""
    if model_type not in ("vit_b", "vit_l", "vit_h"):
        raise ValueError(f"sam_model_type must be vit_b|vit_l|vit_h, got {model_type!r}")
    if model_type == "vit_b" and checkpoint is None:
        return SamEncoderSpec(
            model_type="vit_b", backend="hf",
            hf_id="facebook/sam-vit-base", checkpoint=None,
            encoder_id="facebook/sam-vit-base/vision_encoder",
        )
    ckpt = checkpoint or _DEFAULT_ORIGINAL_CKPT.get(model_type)
    if ckpt is None:
        raise ValueError(
            f"SAM {model_type} needs an original-format checkpoint (.pth); "
            f"pass checkpoint=... (no default registered for {model_type})."
        )
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"SAM {model_type} checkpoint not found: {ckpt}")
    return SamEncoderSpec(
        model_type=model_type, backend="original",
        hf_id=None, checkpoint=ckpt,
        encoder_id=f"segment_anything/{model_type}/image_encoder",
    )


def default_batch_size(spec: SamEncoderSpec) -> int:
    """Per-model batch, sized to FREE GPU memory. ViT-H @1024 rel-pos attention is
    memory-heavy (~3.5GB/sample), so batch 8 needs ~28GB (OK on A40/H100, OOMs a
    32GB V100). Cap by free mem so runtime cache-misses never OOM."""
    base = {"vit_h": 8, "vit_l": 12}.get(spec.model_type, 16)
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if spec.model_type == "vit_h":
                return 8 if free_gb >= 40 else (4 if free_gb >= 22 else 2)
    except Exception:
        pass
    return base


# ----- SAM encoder loader (cached per encoder_id) --------------------------

_LOADED: dict[str, torch.nn.Module] = {}


def _get_encoder(spec: SamEncoderSpec, device: str) -> torch.nn.Module:
    key = spec.encoder_id
    if key not in _LOADED:
        if spec.backend == "hf":
            os.environ.setdefault("HF_HOME", "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub")
            os.environ.setdefault("HF_HUB_CACHE", "/groups/echambe2/gmeng/MedAL-Agent/cache/hf_hub")
            from transformers import SamModel
            _LOADED[key] = SamModel.from_pretrained(spec.hf_id).vision_encoder
        else:
            from segment_anything import sam_model_registry
            sam = sam_model_registry[spec.model_type](checkpoint=spec.checkpoint)
            _LOADED[key] = sam.image_encoder
    enc = _LOADED[key]
    enc.to(device).eval()
    return enc


# ----- preprocess one CHW image to SAM input -------------------------------

def _to_sam_input(img: np.ndarray, cfg: SamPreprocessConfig) -> torch.Tensor:
    """(C, H, W) float32 in [0, 1] -> (1, 3, target, target) float32 normalized."""
    t = torch.from_numpy(img).float()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] != 3:
        raise ValueError(f"SAM features: image must be 1- or 3-channel, got {tuple(t.shape)}")
    t = t.unsqueeze(0)                                          # (1, 3, H, W)
    t = F.interpolate(t, size=(cfg.target_size, cfg.target_size),
                      mode=cfg.resize_mode, align_corners=False)
    t = t * cfg.input_scale_max                                 # [0,1] -> [0,255]
    mean = torch.tensor(cfg.pixel_mean).view(1, 3, 1, 1)
    std  = torch.tensor(cfg.pixel_std).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t


# ----- bulk feature extraction with HDF5 cache -----------------------------

@torch.no_grad()
def _compute_features(
    sample_ids: list[str],
    images: list[np.ndarray],
    *,
    spec: SamEncoderSpec,
    cfg: SamPreprocessConfig,
    device: str,
    batch_size: int,
) -> np.ndarray:
    enc = _get_encoder(spec, device)
    enc.eval()
    out = np.empty((len(sample_ids), spec.feature_dim), dtype=np.float32)
    # inference_mode: no autograd graph -> far less memory (allows big batches) + faster.
    with torch.inference_mode():
        for start in range(0, len(sample_ids), batch_size):
            end = min(start + batch_size, len(sample_ids))
            batch = torch.cat([_to_sam_input(images[i], cfg) for i in range(start, end)], dim=0)
            batch = batch.to(device)
            result = enc(batch)
            # HF vision_encoder -> SamVisionEncoderOutput(.last_hidden_state);
            # original ImageEncoderViT -> a (B, 256, 64, 64) tensor.
            feat = result.last_hidden_state if hasattr(result, "last_hidden_state") else result
            pooled = feat.mean(dim=(-2, -1))                        # (B, 256)
            out[start:end] = pooled.cpu().numpy().astype(np.float32)
    return out


def _cache_path(cache_dir: str, dataset_name: str, spec: SamEncoderSpec,
                cfg: SamPreprocessConfig, in_hw: tuple[int, int] = (0, 0)) -> Path:
    eid = spec.encoder_id.replace("/", "_")
    h = cfg.hash()[:10]
    # in_hw = task input resolution (H, W) feeding SAM; keeps e.g. 128-square and
    # 512-letterbox features in separate files (NOT interchangeable). Both dims are in
    # the key so non-square inputs (H != W) never collide on width alone.
    ih, iw = in_hw
    res = f"__in{ih}x{iw}" if (ih or iw) else ""
    return Path(cache_dir) / f"{dataset_name}__{eid}__{h}{res}.h5"


def _read_cache(path: Path, spec: SamEncoderSpec, cfg: SamPreprocessConfig) -> dict[str, np.ndarray]:
    """Return {sample_id: feature_vector} if cache exists AND its attrs match,
    else {} (caller will recompute)."""
    if not path.exists():
        return {}
    try:
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
            if (attrs.get("encoder_id") != spec.encoder_id or
                attrs.get("preprocess_hash") != cfg.hash() or
                attrs.get("cache_version") != spec.cache_version or
                attrs.get("checkpoint") != (spec.checkpoint or spec.hf_id or "")):
                return {}
            sids = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in f["sample_ids"][:]]
            feats = f["features"][:]
            return {sid: feats[i] for i, sid in enumerate(sids)}
    except (OSError, KeyError):
        return {}


def _write_cache(path: Path, spec: SamEncoderSpec, cfg: SamPreprocessConfig,
                 store: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sids = sorted(store.keys())
    arr = np.stack([store[s] for s in sids], axis=0)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["encoder_id"] = spec.encoder_id
        f.attrs["model_type"] = spec.model_type
        f.attrs["checkpoint"] = spec.checkpoint or spec.hf_id or ""
        f.attrs["preprocess_hash"] = cfg.hash()
        f.attrs["cache_version"] = spec.cache_version
        f.attrs["dim"] = spec.feature_dim
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("sample_ids", data=np.array(sids, dtype=dt))
        f.create_dataset("features", data=arr.astype(np.float32))
    os.replace(tmp, path)


def extract_sam_features(
    ds: MedALDataset,
    *,
    cache_dir: str,
    spec: Optional[SamEncoderSpec] = None,
    device: str = "cuda:0",
    batch_size: Optional[int] = None,
    cfg: Optional[SamPreprocessConfig] = None,
) -> np.ndarray:
    """Return (N, FEATURE_DIM) SAM features for every sample in ``ds`` in
    iteration order. Reads/writes HDF5 cache under ``cache_dir``."""
    spec = spec or resolve_sam_spec()
    cfg = cfg or SamPreprocessConfig()
    batch_size = batch_size or default_batch_size(spec)
    in_hw = (int(ds[0].image.shape[-2]), int(ds[0].image.shape[-1])) if len(ds) else (0, 0)
    path = _cache_path(cache_dir, ds.name, spec, cfg, in_hw=in_hw)
    store = _read_cache(path, spec, cfg)
    sids = ds.sample_ids()
    missing = [(i, sid) for i, sid in enumerate(sids) if sid not in store]
    if missing:
        miss_idx = [i for (i, _) in missing]
        miss_sids = [s for (_, s) in missing]
        images = [ds[i].image for i in miss_idx]
        new_feats = _compute_features(
            miss_sids, images, spec=spec, cfg=cfg, device=device, batch_size=batch_size,
        )
        for j, sid in enumerate(miss_sids):
            store[sid] = new_feats[j]
        _write_cache(path, spec, cfg, store)
    out = np.stack([store[sid] for sid in sids], axis=0)
    return out


def sam_features_meta(spec: Optional[SamEncoderSpec] = None) -> dict:
    spec = spec or resolve_sam_spec()
    return {
        "encoder_id":    spec.encoder_id,
        "model_type":    spec.model_type,
        "backend":       spec.backend,
        "checkpoint":    spec.checkpoint or spec.hf_id or "",
        "layer":         f"{spec.encoder_id} (last_hidden_state)",
        "pooling_rule":  SAM_POOLING_RULE,
        "cache_version": spec.cache_version,
        "dim":           spec.feature_dim,
    }


def make_sam_foundation_fn(cache_dir: str, model_type: str = "vit_b",
                           checkpoint: Optional[str] = None):
    """Factory: return a function with the signature al_loop expects.

    ``model_type`` selects SAM-B (HF) / SAM-L / SAM-H (original .pth)."""
    spec = resolve_sam_spec(model_type, checkpoint)
    meta = sam_features_meta(spec)

    def _fn(*, unlabeled_ds, labeled_ds, seed, device):
        del seed  # SAM features are deterministic on the input pixels
        feats = {
            "foundation_pool":  extract_sam_features(unlabeled_ds, cache_dir=cache_dir, spec=spec, device=device),
            "foundation_label": extract_sam_features(labeled_ds,   cache_dir=cache_dir, spec=spec, device=device),
        }
        return feats, meta
    return _fn
