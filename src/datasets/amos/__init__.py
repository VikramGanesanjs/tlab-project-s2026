"""AMOS self-supervised slice datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AMOSPairedSliceDataset",
    "AMOSSingleSliceDataset",
    "AMOSSliceDataset",
    "DEFAULT_ROOT",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
