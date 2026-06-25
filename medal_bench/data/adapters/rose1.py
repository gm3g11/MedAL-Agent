"""ROSE-1 (retinal OCTA vessel segmentation) adapter — DATA MISSING.

The extracted/ and raw/ directories at the expected root are empty on disk
as of the current pilot setup. Trying to instantiate this adapter raises
FileNotFoundError with the missing path so the smoke test fails loudly
(rather than the pilot training silently skipping this dataset).

When the data is downloaded, the expected layout (ROSE-1 SVC fold) is:
  extracted/<some_release>/{img,gt}/<patient_id>_<eye>.tif  (or .png)

To enable the adapter, populate the directory and replace the body of
__init__ with the standard image+mask glob.
"""
from __future__ import annotations

from pathlib import Path

from medal_bench.data.base import MedALDataset, Sample


class ROSE1Adapter(MedALDataset):
    name = "rose1"
    modality = "octa"
    target = "retinal_vessel"
    dim = "2d"
    query_unit = "image"
    num_classes = 2

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "extracted"
        # Empty / missing -> raise. Any file inside makes us refuse-by-stub too,
        # because the on-disk format will need to be wired up.
        files = list(root.glob("**/*")) if root.exists() else []
        raise FileNotFoundError(
            f"ROSE-1 data not yet downloaded under {root} "
            f"(found {len(files)} entries). Adapter is a stub — see "
            "medal_bench/data/adapters/rose1.py for the expected layout."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def sample_ids(self) -> list[str]:
        raise NotImplementedError

    def __getitem__(self, i: int) -> Sample:
        raise NotImplementedError
