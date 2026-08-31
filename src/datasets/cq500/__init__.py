"""CQ500 CT conversion and PyTorch dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CQ500MultiSliceDataset",
    "CQ500PairedSliceDataset",
    "CQ500SliceDataset",
    "CQ500TaskSpec",
    "CQ500_TASK_CHOICES",
    "DEFAULT_ROOT",
    "build_cq500_transform",
    "build_cq500_volume_transform",
    "convert_cq500_dataset",
    "convert_series_to_nifti",
    "discover_series",
    "resolve_cq500_task",
]

_DATASET_ATTRS = {
    "CQ500MultiSliceDataset",
    "CQ500PairedSliceDataset",
    "CQ500SliceDataset",
    "CQ500TaskSpec",
    "CQ500_TASK_CHOICES",
    "DEFAULT_ROOT",
    "build_cq500_transform",
    "build_cq500_volume_transform",
    "resolve_cq500_task",
}
_CONVERT_ATTRS = {"convert_cq500_dataset", "convert_series_to_nifti", "discover_series"}


def __getattr__(name: str) -> Any:
    if name in _DATASET_ATTRS:
        from . import dataset

        return getattr(dataset, name)
    if name in _CONVERT_ATTRS:
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
