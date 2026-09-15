"""PyTorch datasets for LLD-MMRI liver cancer MRI classification.

Only one acquisition phase is used per dataset instance.  Labels come from
``LLD_MMRI_Annotation.json``; segmentation labels under ``labels/`` are not
used by this module.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset

from datasets.adni.dataset import (
    _VolumeCache,
    _apply_shared_pair_transform,
    _canonical_depth,
    _load_canonical_volume,
    _resample_volume,
    _scale_volume_to_unit_interval,
    _slice_to_pil,
    _volume_to_imagenet_tensors,
    _z_index_range,
    build_adni_transform,
    build_adni_volume_transform,
)
from utils.data_pipeline_timing import DataPipelineTimingProxy

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPOSITORY_ROOT / "data" / "lld-mmri"
DEFAULT_ANNOTATION_NAME = "LLD_MMRI_Annotation.json"
LLD_MMRI_CLASS_NAMES = tuple(f"category_{label}" for label in range(7))

# Public spellings follow the dataset documentation.  The values are the file
# suffixes used under images/ and the normalized annotation phase names.
SCAN_TYPE_CHOICES = ("pre", "+A", "+Delay", "+V", "DWI", "InPhase", "OutPhase", "T2WI")
_SCAN_TYPES = {
    "pre": ("C-pre", "C-pre"),
    "+a": ("C+A", "C+A"),
    "+delay": ("C+Delay", "C+Delay"),
    "+v": ("C+V", "C+V"),
    "dwi": ("DWI", "DWI"),
    "inphase": ("InPhase", "In Phase"),
    "outphase": ("OutPhase", "Out Phase"),
    "t2wi": ("T2WI", "T2WI"),
}
_SCAN_TYPE_ALIASES = {
    "c-pre": "pre", "c+a": "+a", "c+delay": "+delay", "c+v": "+v",
    "in phase": "inphase", "out phase": "outphase",
}


def resolve_scan_type(scan_type: str) -> Tuple[str, str]:
    """Resolve a documented or on-disk scan phase to file and JSON names."""
    key = str(scan_type).strip().lower()
    key = _SCAN_TYPE_ALIASES.get(key, key)
    try:
        return _SCAN_TYPES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scan_type={scan_type!r}; choose from {SCAN_TYPE_CHOICES}"
        ) from exc


def resolve_scan_types(
    scan_type: Union[str, Sequence[str]],
) -> Tuple[Tuple[str, str], ...]:
    """Resolve one or more scan phases, or ``"all"`` for every phase.

    A comma-separated string is accepted for command-line convenience, while
    YAML and direct callers may supply a sequence such as ``["pre", "DWI"]``.
    """
    if isinstance(scan_type, str):
        values: Sequence[str] = (
            tuple(part.strip() for part in scan_type.split(","))
            if "," in scan_type
            else (scan_type,)
        )
    elif isinstance(scan_type, Sequence):
        values = scan_type
    else:
        raise TypeError(
            "scan_type must be a scan name, a sequence of scan names, or 'all'; "
            f"got {type(scan_type).__name__}"
        )
    if not values:
        raise ValueError("scan_type must select at least one phase")
    if any(str(value).strip().lower() == "all" for value in values):
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with individual scan types")
        return tuple(_SCAN_TYPES[key] for key in _SCAN_TYPES)

    resolved = tuple(resolve_scan_type(str(value)) for value in values)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"scan_type contains duplicate phases: {scan_type!r}")
    return resolved


@dataclass(frozen=True)
class _ScanRecord:
    image_id: str
    patient_id: str
    volume_path: Path
    n_slices: int
    label: int
    phenotype: Dict[str, str]


def _case_id_from_image_name(name: str, file_phase: str) -> str:
    suffix = f"_{file_phase}_0000.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"Image filename does not end in expected phase suffix {suffix!r}: {name}")
    image_id = name[: -len(suffix)]
    # The trailing number differentiates lesion/image files; JSON keys use the
    # MRI case identifier preceding it (for example MR-398189_1 -> MR-398189).
    patient_id, separator, _ = image_id.rpartition("_")
    if not separator or not patient_id:
        raise ValueError(f"Could not extract MRI case ID from image filename {name!r}")
    return patient_id


def _read_annotations(annotation_path: Path, annotation_phase: str) -> Dict[str, int]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing LLD-MMRI annotation JSON: {annotation_path}")
    try:
        with annotation_path.open(encoding="utf-8") as handle:
            cases = json.load(handle)["Annotation_info"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid LLD-MMRI annotation JSON {annotation_path}: {exc}") from exc
    if not isinstance(cases, dict):
        raise ValueError("LLD-MMRI Annotation_info must be a mapping")

    labels: Dict[str, int] = {}
    for patient_id, entries in cases.items():
        if not isinstance(entries, list):
            raise ValueError(f"Invalid annotation entries for MRI case {patient_id!r}")
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("phase") == annotation_phase]
        if len(matches) > 1:
            raise ValueError(f"Multiple {annotation_phase!r} annotations for MRI case {patient_id!r}")
        if not matches:
            continue
        category = matches[0].get("annotation", {}).get("lesion", {}).get("0", {}).get("category")
        if isinstance(category, bool) or not isinstance(category, (int, float)) or int(category) != category:
            raise ValueError(f"Invalid diagnosis category for MRI case {patient_id!r}: {category!r}")
        label = int(category)
        if not 0 <= label <= 6:
            raise ValueError(f"Diagnosis category for MRI case {patient_id!r} must be in [0, 6], got {label}")
        labels[str(patient_id)] = label
    return labels


def _build_scan_index(
    root: Path,
    annotation_path: Path,
    scan_types: Sequence[Tuple[str, str]],
    patient_ids: Optional[Sequence[str]],
    *,
    include_native_depth: bool,
) -> List[_ScanRecord]:
    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing LLD-MMRI images directory: {image_root}")
    requested = {str(patient_id) for patient_id in patient_ids} if patient_ids is not None else None
    records: List[_ScanRecord] = []
    missing_annotations: List[str] = []
    for file_phase, annotation_phase in scan_types:
        labels = _read_annotations(annotation_path, annotation_phase)
        suffix = f"_{file_phase}_0000.nii.gz"
        for path in sorted(image_root.glob(f"*{suffix}")):
            patient_id = _case_id_from_image_name(path.name, file_phase)
            if requested is not None and patient_id not in requested:
                continue
            label = labels.get(patient_id)
            if label is None:
                missing_annotations.append(path.name)
                continue
            image_id = path.name[: -len(".nii.gz")]
            records.append(_ScanRecord(
                image_id=image_id,
                patient_id=patient_id,
                volume_path=path,
                n_slices=_canonical_depth(path) if include_native_depth else 0,
                label=label,
                phenotype={"Category": str(label), "Phase": annotation_phase},
            ))
    if missing_annotations:
        raise ValueError(
            f"{len(missing_annotations)} selected LLD-MMRI image(s) have no matching annotation "
            f"(examples: {missing_annotations[:5]})"
        )
    if requested is not None:
        available = {record.patient_id for record in records}
        missing = requested - available
        if missing:
            raise ValueError(f"Requested MRI case IDs have no selected scan: {sorted(missing)[:5]}")
    if not records:
        raise RuntimeError(f"No selected LLD-MMRI scans found under {image_root}")
    logger.info("Indexed LLD-MMRI phases=%s scans=%d labels=%s", [file_phase for file_phase, _ in scan_types], len(records), {label: sum(r.label == label for r in records) for label in range(7)})
    return records


# These aliases intentionally keep the exact ADNI image normalization and 3-D
# augmentation policy, while making the LLD-MMRI public API self-contained.
build_lld_mmri_transform = build_adni_transform
build_lld_mmri_volume_transform = build_adni_volume_transform


class _LLDMMRIBaseDataset(VisionDataset):
    def __init__(self, root: Union[str, Path] = DEFAULT_ROOT, *, scan_type: Union[str, Sequence[str]], annotation_path: Optional[Union[str, Path]] = None, patient_ids: Optional[Sequence[str]] = None, z_min: float = 0.0, z_max: float = 1.0, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, volume_cache_size: Optional[int] = 8, augment: bool = True, image_size: int = 224, build_default_transform: bool = True, data_timing: Optional[DataPipelineTimingProxy] = None) -> None:
        root_path = Path(root).expanduser().resolve()
        annotation_file = Path(annotation_path).expanduser().resolve() if annotation_path is not None else root_path / DEFAULT_ANNOTATION_NAME
        self.scan_types = resolve_scan_types(scan_type)
        if build_default_transform and transforms is None and transform is None:
            transform = build_lld_mmri_transform(image_size=image_size, augment=augment)
        super().__init__(str(root_path), transforms=transforms, transform=transform, target_transform=target_transform)
        self.root_path = root_path
        self.annotation_path = annotation_file
        self.data_timing = data_timing
        self.z_min, self.z_max = float(z_min), float(z_max)
        _z_index_range(1, self.z_min, self.z_max)
        self._volume_cache = _VolumeCache(volume_cache_size) if volume_cache_size is not None else None
        self._records = _build_scan_index(root_path, annotation_file, self.scan_types, patient_ids, include_native_depth=not isinstance(self, LLDMMRIMultiSliceDataset))

    @staticmethod
    def _record_metadata(record: _ScanRecord) -> Dict[str, str]:
        return dict(record.phenotype)


class LLDMMRIClassificationDataset(_LLDMMRIBaseDataset):
    """Single axial-slice LLD-MMRI classification dataset (diagnosis labels 0–6)."""
    def __init__(self, root: Union[str, Path] = DEFAULT_ROOT, *, scan_type: Union[str, Sequence[str]], volume_cache_size: int = 8, **kwargs: Any) -> None:
        super().__init__(root=root, scan_type=scan_type, volume_cache_size=volume_cache_size, **kwargs)
        self._entries: List[Tuple[_ScanRecord, int]] = []
        for record in self._records:
            start, end = _z_index_range(record.n_slices, self.z_min, self.z_max)
            self._entries.extend((record, z) for z in range(start, end))
        if not self._entries:
            raise RuntimeError(f"No LLD-MMRI slices found under {self.root_path} in z-range [{self.z_min}, {self.z_max})")

    def __len__(self) -> int: return len(self._entries)
    def get_target(self, index: int) -> int: return self._entries[index][0].label
    def get_patient_id(self, index: int) -> str: return self._entries[index][0].patient_id
    def get_image_id(self, index: int) -> str: return self._entries[index][0].image_id
    def get_phenotype_raw(self, index: int) -> Dict[str, str]: return self._record_metadata(self._entries[index][0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        assert self._volume_cache is not None
        image: Any = _slice_to_pil(self._volume_cache.get(record.volume_path), z)
        target: Any = record.label
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None: image = self.transform(image)
            if self.target_transform is not None: target = self.target_transform(target)
        return image, target


class LLDMMRIMultiSliceDataset(_LLDMMRIBaseDataset):
    """Scan-level LLD-MMRI dataset with ADNI-identical volume preprocessing."""
    def __init__(self, root: Union[str, Path] = DEFAULT_ROOT, *, scan_type: Union[str, Sequence[str]], n_slices: int = 8, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, augment: bool = True, image_size: int = 224, return_imagenet_tensors: bool = True, three_d_encoder: bool = False, data_timing: Optional[DataPipelineTimingProxy] = None, **kwargs: Any) -> None:
        if n_slices <= 0: raise ValueError(f"n_slices must be positive, got {n_slices}")
        if image_size <= 0: raise ValueError(f"image_size must be positive, got {image_size}")
        self.n_slices, self.image_size = int(n_slices), int(image_size)
        self.return_imagenet_tensors, self.three_d_encoder = bool(return_imagenet_tensors), bool(three_d_encoder)
        if transforms is None and transform is None and augment: transform = build_lld_mmri_volume_transform(augment=True)
        super().__init__(root=root, scan_type=scan_type, transforms=transforms, transform=transform, target_transform=target_transform, volume_cache_size=None, augment=augment, image_size=image_size, build_default_transform=False, data_timing=data_timing, **kwargs)
        self._entries = list(self._records)

    def __len__(self) -> int: return len(self._entries)
    def get_target(self, index: int) -> int: return self._entries[index].label
    def get_patient_id(self, index: int) -> str: return self._entries[index].patient_id
    def get_image_id(self, index: int) -> str: return self._entries[index].image_id
    def get_phenotype_raw(self, index: int) -> Dict[str, str]: return self._record_metadata(self._entries[index])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record = self._entries[index]
        volume = _load_canonical_volume(record.volume_path)
        volume_t = _resample_volume(_scale_volume_to_unit_interval(volume), self.n_slices, self.image_size).unsqueeze(0)
        target: Any = record.label
        if self.transforms is not None: volume_t, target = self.transforms(volume_t, target)
        else:
            if self.transform is not None: volume_t = self.transform(volume_t)
            if self.target_transform is not None: target = self.target_transform(target)
        volume_t = torch.as_tensor(volume_t)
        if volume_t.ndim == 3: volume_t = volume_t.unsqueeze(0)
        if volume_t.ndim != 4 or volume_t.shape[0] != 1:
            raise ValueError(f"LLD-MMRI multi-slice transforms must return [1, depth, height, width], got shape {tuple(volume_t.shape)}")
        if self.three_d_encoder: images = volume_t.permute(1, 0, 2, 3)
        elif self.return_imagenet_tensors: images = _volume_to_imagenet_tensors(volume_t.squeeze(0))
        else: images = volume_t
        return images, target


class LLDMMRIPairedSliceDataset(LLDMMRIClassificationDataset):
    """Paired nearby axial slices from the same selected LLD-MMRI phase."""
    def __init__(self, root: Union[str, Path] = DEFAULT_ROOT, *, scan_type: Union[str, Sequence[str]], max_distance: int = 3, seed: Optional[int] = None, **kwargs: Any) -> None:
        if max_distance < 0: raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        self.max_distance, self._seed = int(max_distance), seed
        super().__init__(root=root, scan_type=scan_type, **kwargs)

    def _rng_for_index(self, index: int) -> np.random.RandomState:
        return np.random.RandomState() if self._seed is None else np.random.RandomState(self._seed + int(index))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        start, end = _z_index_range(record.n_slices, self.z_min, self.z_max)
        candidates = [candidate for candidate in range(max(start, z - self.max_distance), min(end - 1, z + self.max_distance) + 1) if candidate != z]
        partner_z = int(self._rng_for_index(index).choice(candidates)) if candidates else z
        assert self._volume_cache is not None
        volume = self._volume_cache.get(record.volume_path)
        image: Any = (_slice_to_pil(volume, z), _slice_to_pil(volume, partner_z))
        if self.transform is not None: image = _apply_shared_pair_transform(self.transform, image)
        elif self.transforms is not None:
            transformed = self.transforms(image, None)
            image = transformed[0] if isinstance(transformed, tuple) else transformed
        return image


__all__ = ["LLDMMRIClassificationDataset", "LLDMMRIMultiSliceDataset", "LLDMMRIPairedSliceDataset", "DEFAULT_ANNOTATION_NAME", "DEFAULT_ROOT", "LLD_MMRI_CLASS_NAMES", "SCAN_TYPE_CHOICES", "build_lld_mmri_transform", "build_lld_mmri_volume_transform", "resolve_scan_type", "resolve_scan_types"]
