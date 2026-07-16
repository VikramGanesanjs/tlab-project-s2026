"""Duke Breast Cancer MRI helpers for conversion and DINOv3-compatible datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CANONICAL_SCAN_TYPES",
    "DEFAULT_PHENOTYPE_COLUMNS",
    "PHENOTYPE_SENTINEL",
    "DukeBreastMRIDataset",
    "PairToDinoGlobalCrops",
    "build_phenotype_json",
    "convert_duke_dataset",
    "convert_series_to_nifti",
    "resolve_scan",
]

_DATASET_ATTRS = {"DukeBreastMRIDataset", "PairToDinoGlobalCrops"}
_CONVERT_ATTRS = {
    "CANONICAL_SCAN_TYPES",
    "DEFAULT_PHENOTYPE_COLUMNS",
    "PHENOTYPE_SENTINEL",
    "build_phenotype_json",
    "convert_duke_dataset",
    "convert_series_to_nifti",
    "resolve_scan",
}


def __getattr__(name: str) -> Any:
    if name in _DATASET_ATTRS:
        from . import dataset

        return getattr(dataset, name)
    if name in _CONVERT_ATTRS:
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
