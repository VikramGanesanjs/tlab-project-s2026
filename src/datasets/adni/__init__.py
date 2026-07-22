"""ADNI1 1.5T MRI dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ADNIClassificationDataset",
    "ADNIMultiSliceDataset",
    "ADNIPairedSliceDataset",
    "DEFAULT_CSV_NAME",
    "DEFAULT_PHENOTYPE_COLUMNS",
    "DEFAULT_ROOT",
    "DIAGNOSIS_TO_LABEL",
    "LABEL_TO_DIAGNOSIS",
    "PHENOTYPE_SENTINEL",
    "build_adni_transform",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
