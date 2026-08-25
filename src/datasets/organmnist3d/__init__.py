"""OrganMNIST3D dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_NPZ_NAME",
    "DEFAULT_ROOT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "OrganMNIST3DClassificationDataset",
    "OrganMNIST3DMultiSliceDataset",
    "OrganMNIST3DPairedSliceDataset",
    "ORGANMNIST3D_CLASS_NAMES",
    "SPLIT_CHOICES",
    "build_organmnist3d_transform",
    "build_organmnist3d_volume_transform",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
