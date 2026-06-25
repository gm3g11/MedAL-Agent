"""Dense label-remapping for datasets with non-contiguous native label codes.

Many segmentation datasets store masks with sparse / high-valued native codes
(e.g. MMWHS uses {0,205,420,...,850}; BTCV uses {0..12,16}). The training loss
and metrics need dense class indices {0..C-1}. ``LabelRemapper`` applies that
mapping **inside the adapter's __getitem__ (the exact training-loader path)** and
hard-errors on any native code it does not know — so an unexpected label can
never be silently mislabeled.

Read masks as integer volumes (nibabel) BEFORE remapping; never route a mask
with codes > 255 through an 8-bit image decoder (it would wrap/collide).
"""
from __future__ import annotations

import numpy as np


class LabelRemapper:
    """Vectorized native-code -> dense-class remap with unknown-code guard."""

    def __init__(self, mapping: dict[int, int], name: str):
        if 0 not in mapping or mapping[0] != 0:
            raise ValueError(f"[{name}] remap must preserve background 0->0; got {mapping.get(0)}")
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.name = name
        self._max_code = max(self.mapping)
        # LUT indexed by native code; -1 marks an unmapped (unknown) code.
        lut = np.full(self._max_code + 1, -1, dtype=np.int64)
        for native, dense in self.mapping.items():
            lut[native] = dense
        self._lut = lut

    @property
    def num_classes(self) -> int:
        return max(self.mapping.values()) + 1

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Return a dense int64 mask in {0..num_classes-1}; raise on unknown codes."""
        m = np.asarray(mask)
        # masks may arrive as float (e.g. BTCV float32 NIfTI); round to nearest int.
        if not np.issubdtype(m.dtype, np.integer):
            m = np.rint(m).astype(np.int64)
        else:
            m = m.astype(np.int64)
        lo, hi = int(m.min()), int(m.max())
        if lo < 0 or hi > self._max_code:
            bad = sorted(set(np.unique(m).tolist()) - set(self.mapping))
            raise ValueError(
                f"[{self.name}] native label codes {bad} are outside the remap "
                f"{sorted(self.mapping)} (mask range [{lo},{hi}])"
            )
        out = self._lut[m]
        if (out < 0).any():
            bad = sorted({int(c) for c in np.unique(m[out < 0])})
            raise ValueError(
                f"[{self.name}] unknown native label codes {bad}; "
                f"remap knows {sorted(self.mapping)}"
            )
        return out


# --- named remaps (verified against on-disk label volumes 2026-06-13) ---

# MMWHS whole-heart: 8 classes. 421 (extra blood-pool code, only in Case3010) -> 2.
MMWHS_REMAP = {0: 0, 205: 1, 420: 2, 421: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}

# BTCV/Synapse abdominal multi-organ: native {0..12,16}; 13,14,15 absent -> 16->13.
# Yields 14 dense classes (0=bg + 13 organs). Confirmed by full-volume scan.
BTCV_REMAP = {**{i: i for i in range(13)}, 16: 13}

# Prepared for future wiring (not yet registered) ---------------------------
# MyOPS cardiac pathology: 5 classes (1220 & 2221 both -> scar/edema class 4).
MYOPS_REMAP = {0: 0, 200: 1, 500: 2, 600: 3, 1220: 4, 2221: 4}
# BraTS brain tumor: native {0,1,2,4} -> {0,1,2,3} (4 classes; no unused class 3).
BRATS_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}
