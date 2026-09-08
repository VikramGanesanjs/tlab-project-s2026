"""ADNI1 1.5T MRI dataset helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ADNIClassificationDataset",
    "ADNIMultiSliceDataset",
    "ADNIPairedSliceDataset",
    "ADNITaskSpec",
    "ADNI_TASK_CHOICES",
    "DEFAULT_ADNI_TASK",
    "DEFAULT_CSV_NAME",
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_PHENOTYPE_COLUMNS",
    "DEFAULT_ROOT",
    "DIAGNOSIS_TO_LABEL",
    "LABEL_TO_DIAGNOSIS",
    "PHENOTYPE_SENTINEL",
    "build_adni_transform",
    "build_adni_volume_transform",
    "resolve_adni_task",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
