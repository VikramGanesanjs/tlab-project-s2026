"""LLD-MMRI liver cancer classification dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LLDMMRIClassificationDataset",
    "LLDMMRIMultiSliceDataset",
    "LLDMMRIPairedSliceDataset",
    "DEFAULT_ANNOTATION_NAME",
    "DEFAULT_ROOT",
    "LLD_MMRI_CLASS_NAMES",
    "SCAN_TYPE_CHOICES",
    "build_lld_mmri_transform",
    "build_lld_mmri_volume_transform",
    "resolve_scan_type",
    "resolve_scan_types",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
