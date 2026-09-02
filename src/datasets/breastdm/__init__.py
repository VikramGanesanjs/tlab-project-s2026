"""BreastDM DCE-MRI dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BREASTDM_CLASS_NAMES",
    "DEFAULT_ROOT",
    "SPLIT_CHOICES",
    "BreastDMMultiSliceDataset",
    "BreastDMPairedSliceDataset",
    "BreastDMSingleSliceDataset",
    "BreastDMSliceDataset",
    "build_breastdm_transform",
    "build_breastdm_volume_transform",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
