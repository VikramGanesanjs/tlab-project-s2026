"""PyTorch datasets for the ADNI1 1.5T MRI collection.

The source dataset is consumed in place: NIfTI files are matched to the ADNI
CSV by the image ID stored in each file's direct parent directory.  No
conversion or segmentation products are required.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

from utils.data_pipeline_timing import DataPipelineTimingProxy

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPOSITORY_ROOT / "data" / "ADNI"
DEFAULT_CSV_NAME = "ADNI1_Complete_3Yr_1.5T_7_21_2026.csv"
DEFAULT_MANIFEST_NAME = "adni_nii_manifest.json"

DIAGNOSIS_TO_LABEL = {"CN": 0, "MCI": 1, "AD": 2}
LABEL_TO_DIAGNOSIS = {label: diagnosis for diagnosis, label in DIAGNOSIS_TO_LABEL.items()}
DEFAULT_PHENOTYPE_COLUMNS = ("Group", "Sex", "Age")
PHENOTYPE_SENTINEL = -1.0

# Multi-slice / classification task configurations.
# Binary tasks use Duke-style labels (0/1) with the second diagnosis positive.
ADNI_TASK_CHOICES = ("cn_mci_ad", "cn_ad", "cn_mci", "mci_ad")
DEFAULT_ADNI_TASK = "cn_mci_ad"


@dataclass(frozen=True)
class ADNITaskSpec:
    """Resolved ADNI classification task: diagnoses kept and label remapping."""

    name: str
    diagnoses: Tuple[str, ...]
    label_map: Dict[str, int]
    class_names: Tuple[str, ...]
    binary: bool

    @property
    def num_logits(self) -> int:
        """Classifier output width: 1 for BCE binary, else the class count."""
        return 1 if self.binary else len(self.class_names)


_ADNI_TASK_SPECS: Dict[str, ADNITaskSpec] = {
    "cn_mci_ad": ADNITaskSpec(
        name="cn_mci_ad",
        diagnoses=("CN", "MCI", "AD"),
        label_map={"CN": 0, "MCI": 1, "AD": 2},
        class_names=("CN", "MCI", "AD"),
        binary=False,
    ),
    "cn_ad": ADNITaskSpec(
        name="cn_ad",
        diagnoses=("CN", "AD"),
        label_map={"CN": 0, "AD": 1},
        class_names=("CN", "AD"),
        binary=True,
    ),
    "cn_mci": ADNITaskSpec(
        name="cn_mci",
        diagnoses=("CN", "MCI"),
        label_map={"CN": 0, "MCI": 1},
        class_names=("CN", "MCI"),
        binary=True,
    ),
    "mci_ad": ADNITaskSpec(
        name="mci_ad",
        diagnoses=("MCI", "AD"),
        label_map={"MCI": 0, "AD": 1},
        class_names=("MCI", "AD"),
        binary=True,
    ),
}


def resolve_adni_task(task: str) -> ADNITaskSpec:
    """Return the task spec for an ADNI classification configuration."""
    normalized = str(task).strip().lower().replace("-", "_")
    spec = _ADNI_TASK_SPECS.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unknown ADNI task={task!r}; choose from {ADNI_TASK_CHOICES}"
        )
    return spec

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_ID_RE = re.compile(r"^I[0-9]+$")


def _apply_shared_pair_transform(transform: Callable, images: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """Apply identical stochastic augmentation parameters to both images."""
    torch_initial = torch.get_rng_state()
    numpy_initial = np.random.get_state()
    python_initial = random.getstate()

    first = transform(images[0])
    torch_advanced = torch.get_rng_state()
    numpy_advanced = np.random.get_state()
    python_advanced = random.getstate()

    torch.set_rng_state(torch_initial)
    np.random.set_state(numpy_initial)
    random.setstate(python_initial)
    second = transform(images[1])

    torch.set_rng_state(torch_advanced)
    np.random.set_state(numpy_advanced)
    random.setstate(python_advanced)
    return first, second


@dataclass(frozen=True)
class _ScanRecord:
    image_id: str
    patient_id: str
    volume_path: Path
    n_slices: int
    label: int
    phenotype: Dict[str, str]


class _VolumeCache:
    """Process-local LRU of closest-canonical, mmap-derived NIfTI arrays."""

    def __init__(self, maxsize: int = 8) -> None:
        if maxsize <= 0:
            raise ValueError(f"volume_cache_size must be positive, got {maxsize}")
        self.maxsize = int(maxsize)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        volume = _load_canonical_volume(path)

        self._cache[key] = volume
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return volume


def _load_canonical_volume(path: Path) -> np.ndarray:
    """Load a canonical ADNI volume without retaining it between samples."""
    import nibabel as nib

    image = nib.load(str(path), mmap="r")
    canonical = nib.as_closest_canonical(image)
    volume = np.asanyarray(canonical.dataobj)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}, got shape {volume.shape}")
    return volume


def _canonical_depth(path: Path) -> int:
    """Read the canonical superior/inferior depth without loading voxel data."""
    import nibabel as nib

    image = nib.load(str(path), mmap="r")
    orientation = nib.orientations.io_orientation(image.affine)
    canonical_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orientation, canonical_orientation)
    canonical_shape = tuple(int(image.shape[int(source_axis)]) for source_axis in transform[:, 0])
    if len(canonical_shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}, got shape {image.shape}")
    return canonical_shape[-1]


def _normalize_image_id(value: Any) -> str:
    image_id = str(value).strip().upper()
    if not _IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError(f"Invalid ADNI image ID {value!r}; expected a value like 'I33480'")
    return image_id


def _manifest_path(root: Path, manifest_path: Optional[Union[str, Path]]) -> Path:
    return (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root / DEFAULT_MANIFEST_NAME
    )


def _manifest_volumes(
    root: Path,
    manifest_path: Path,
    patient_ids: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """Load indexed paths, reading only the requested patient entries."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing ADNI NIfTI manifest: {manifest_path}. Create it once before "
            "training with `python -m datasets.adni.build_manifest --root "
            f"{root}`."
        )
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        patients = manifest["patients"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid ADNI NIfTI manifest {manifest_path}: {exc}") from exc
    if not isinstance(patients, dict):
        raise ValueError(f"Invalid ADNI NIfTI manifest {manifest_path}: patients must be a mapping")
    requested = {str(patient_id) for patient_id in patient_ids} if patient_ids is not None else set(patients)
    missing = requested.difference(patients)
    if missing:
        raise ValueError(
            f"ADNI manifest {manifest_path} is missing requested patient IDs: {sorted(missing)[:5]}"
        )
    volumes: Dict[str, Path] = {}
    for patient_id in sorted(requested):
        entries = patients[patient_id]
        if not isinstance(entries, list):
            raise ValueError(f"Invalid entries for ADNI patient {patient_id!r} in {manifest_path}")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("image_id"), str) or not isinstance(entry.get("path"), str):
                raise ValueError(f"Invalid NIfTI entry for ADNI patient {patient_id!r} in {manifest_path}")
            image_id = _normalize_image_id(entry["image_id"])
            volume_path = root / entry["path"]
            if image_id in volumes:
                raise ValueError(f"Duplicate image ID {image_id} in ADNI manifest {manifest_path}")
            if not volume_path.is_file():
                raise FileNotFoundError(f"ADNI manifest path does not exist: {volume_path}")
            volumes[image_id] = volume_path
    if not volumes:
        raise RuntimeError(f"No ADNI NIfTI paths selected from {manifest_path}")
    return volumes


def _read_metadata(csv_path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing ADNI metadata CSV: {csv_path}")

    rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        required = {"Image Data ID", "Subject", "Group"}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"ADNI CSV is missing required columns: {sorted(missing)}")

        for line_number, source_row in enumerate(reader, start=2):
            row = {str(key).strip(): str(value or "").strip() for key, value in source_row.items()}
            image_id = _normalize_image_id(row["Image Data ID"])
            if image_id in rows:
                raise ValueError(
                    f"Duplicate image ID {image_id} in {csv_path} at line {line_number}"
                )
            diagnosis = row["Group"].upper()
            if diagnosis not in DIAGNOSIS_TO_LABEL:
                raise ValueError(
                    f"Unknown diagnosis {row['Group']!r} for image ID {image_id}; "
                    f"expected one of {sorted(DIAGNOSIS_TO_LABEL)}"
                )
            row["Image Data ID"] = image_id
            row["Group"] = diagnosis
            rows[image_id] = row

    if not rows:
        raise RuntimeError(f"No metadata rows found in {csv_path}")
    return columns, rows


def _build_scan_index(
    root: Path,
    csv_path: Path,
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    patient_ids: Optional[Sequence[str]] = None,
    include_native_depth: bool = True,
) -> Tuple[List[_ScanRecord], List[str]]:
    volumes = _manifest_volumes(root, _manifest_path(root, manifest_path), patient_ids)
    columns, metadata = _read_metadata(csv_path)

    file_ids = set(volumes)
    metadata_ids = set(metadata)
    missing_files = sorted(metadata_ids - file_ids) if patient_ids is None else []
    missing_metadata = sorted(file_ids - metadata_ids)
    if missing_files or missing_metadata:
        details = []
        if missing_files:
            details.append(
                f"{len(missing_files)} CSV IDs have no file (examples: {missing_files[:5]})"
            )
        if missing_metadata:
            details.append(
                f"{len(missing_metadata)} file IDs have no CSV row "
                f"(examples: {missing_metadata[:5]})"
            )
        raise ValueError("ADNI image/metadata mismatch: " + "; ".join(details))

    records: List[_ScanRecord] = []
    for image_id in sorted(file_ids, key=lambda item: int(item[1:])):
        path = volumes[image_id]
        phenotype = metadata[image_id]
        patient_id = phenotype["Subject"]
        relative_parts = path.relative_to(root).parts
        if not relative_parts or relative_parts[0] != patient_id:
            raise ValueError(
                f"Subject mismatch for {image_id}: CSV has {patient_id!r}, "
                f"but file is under {relative_parts[0] if relative_parts else path!r}"
            )
        records.append(
            _ScanRecord(
                image_id=image_id,
                patient_id=patient_id,
                volume_path=path,
                n_slices=_canonical_depth(path) if include_native_depth else 0,
                label=DIAGNOSIS_TO_LABEL[phenotype["Group"]],
                phenotype=phenotype,
            )
        )

    logger.info(
        "Indexed ADNI root=%s scans=%d patients=%d labels=%s",
        root,
        len(records),
        len({record.patient_id for record in records}),
        {
            diagnosis: sum(record.label == label for record in records)
            for diagnosis, label in DIAGNOSIS_TO_LABEL.items()
        },
    )
    return records, columns


def _z_index_range(n_z: int, z_min: float, z_max: float) -> Tuple[int, int]:
    if n_z <= 0:
        return 0, 0
    if not (0.0 <= z_min <= z_max <= 1.0):
        raise ValueError(
            f"Require 0 <= z_min <= z_max <= 1; got z_min={z_min}, z_max={z_max}"
        )
    start = max(0, min(int(np.floor(n_z * z_min)), n_z))
    end = max(start, min(int(np.ceil(n_z * z_max)), n_z))
    return start, end


def _slice_to_pil(
    volume: np.ndarray,
    z: int,
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Image.Image:
    image_slice = np.asarray(volume[..., z], dtype=np.float32)
    finite = image_slice[np.isfinite(image_slice)]
    if finite.size == 0:
        raise ValueError(f"Axial slice {z} contains no finite values")
    low, high = np.percentile(finite, percentiles)
    if high <= low:
        high = low + 1.0
    normalized = np.nan_to_num(
        np.clip((image_slice - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    pixels = (normalized * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def _scale_volume_to_unit_interval(volume: np.ndarray) -> np.ndarray:
    """Map finite ADNI intensities from the volume's 0th–99th percentile to [0, 1].

    This preserves a single intensity scale for the complete MRI volume.  It is
    deliberately applied before spatial resampling and ImageNet normalization:
    per-slice scaling would erase inter-slice intensity relationships, while a
    z-score would be discarded by the subsequent conversion to image pixels.
    """
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("ADNI volume contains no finite voxel values")
    low = float(finite.min())
    high = float(np.percentile(finite, 99.0))
    if high <= low:
        high = low + 1.0
    return np.nan_to_num(
        np.clip((volume - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def _resample_volume(
    volume: np.ndarray,
    n_slices: int,
    image_size: int,
) -> torch.Tensor:
    """Return a canonical volume as ``[depth, image_size, image_size]``."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    if n_slices <= 0:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    height, width, _ = volume.shape
    volume_t = torch.from_numpy(
        np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    ).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    volume_t = F.interpolate(
        volume_t,
        size=(int(n_slices), height, width),
        mode="trilinear",
        align_corners=False,
    )
    volume_t = F.interpolate(
        volume_t.squeeze(0),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )
    return volume_t.squeeze(0)


def _volume_to_imagenet_tensors(volume: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize volume-level [0, 1] grayscale slices as RGB."""
    slices = torch.as_tensor(volume, dtype=torch.float32)
    if slices.ndim != 3:
        raise ValueError(
            "Expected grayscale slices shaped [depth, height, width], "
            f"got shape {tuple(slices.shape)}"
        )

    images = torch.nan_to_num(slices.clamp(0.0, 1.0), nan=0.0).unsqueeze(1).repeat(
        1, 3, 1, 1
    )
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def build_adni_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.0,
) -> tv_transforms.Compose:
    """Build Duke-compatible preprocessing with conservative brain MRI flips."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if not 0.0 < crop_scale_min <= 1.0:
        raise ValueError(f"crop_scale_min must be in (0, 1], got {crop_scale_min}")
    if jitter < 0.0 or rotation_degrees < 0.0:
        raise ValueError("jitter and rotation_degrees must be non-negative")
    for name, probability in (
        ("horizontal_flip_prob", horizontal_flip_prob),
        ("vertical_flip_prob", vertical_flip_prob),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {probability}")

    operations: List[Callable] = []
    if augment:
        operations.extend(
            [
                tv_transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_scale_min, 1.0),
                    ratio=(0.9, 1.1),
                ),
                tv_transforms.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=min(jitter / 2.0, 0.5),
                ),
                tv_transforms.RandomRotation(rotation_degrees),
                tv_transforms.RandomHorizontalFlip(horizontal_flip_prob),
                tv_transforms.RandomVerticalFlip(vertical_flip_prob),
            ]
        )
    else:
        operations.append(tv_transforms.Resize((image_size, image_size)))
    operations.extend(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tv_transforms.Compose(operations)


def build_adni_volume_transform(*, augment: bool = True) -> Optional[Callable]:
    """Build a MONAI transform for an entire ADNI volume.

    The returned callable accepts a channel-first ``[1, D, H, W]`` tensor and
    returns a transformed tensor with the same layout.  MONAI is imported
    lazily so the single-slice ADNI datasets retain their existing dependency
    requirements when this multi-slice path is not used.
    """
    if not augment:
        return None

    try:
        from monai.transforms import (
            Compose,
            RandAdjustContrastd,
            RandAffined,
            RandFlipd,
            RandGaussianNoised,
            RandGaussianSmoothd,
        )
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for ADNI multi-slice augmentation; "
            "install the project's MONAI dependency or set augment=False"
        ) from exc

    transform = Compose(
        [
            RandAffined(
                keys=("image",),
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                prob=0.5,
                padding_mode="border",
                mode="trilinear",
            ),
            RandFlipd(keys=("image",), spatial_axis=[2], prob=0.5),
            RandGaussianSmoothd(keys=("image",), prob=0.2),
            RandGaussianNoised(keys=("image",), prob=0.2, std=0.05),
            RandAdjustContrastd(keys=("image",), prob=0.2, gamma=(0.7, 1.3)),
        ]
    )

    def apply(volume: torch.Tensor) -> torch.Tensor:
        transformed = transform({"image": volume})
        return transformed["image"]

    return apply


class _ADNIBaseDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        csv_path: Optional[Union[str, Path]] = None,
        task: str = DEFAULT_ADNI_TASK,
        patient_ids: Optional[Sequence[str]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: Optional[int] = 8,
        augment: bool = True,
        image_size: int = 224,
        build_default_transform: bool = True,
        data_timing: Optional[DataPipelineTimingProxy] = None,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        metadata_path = (
            Path(csv_path).expanduser().resolve()
            if csv_path is not None
            else root_path / DEFAULT_CSV_NAME
        )
        if build_default_transform and transforms is None and transform is None:
            transform = build_adni_transform(image_size=image_size, augment=augment)
        super().__init__(
            str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root_path
        self.data_timing = data_timing
        self.csv_path = metadata_path
        self.task_spec = resolve_adni_task(task)
        self.task = self.task_spec.name
        self.class_names = self.task_spec.class_names
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        _z_index_range(1, self.z_min, self.z_max)
        self._volume_cache = (
            _VolumeCache(volume_cache_size) if volume_cache_size is not None else None
        )
        records, self.phenotype_columns = _build_scan_index(
            root_path,
            metadata_path,
            manifest_path=manifest_path,
            patient_ids=patient_ids,
            include_native_depth=not isinstance(self, ADNIMultiSliceDataset),
        )
        allowed = set(self.task_spec.diagnoses)
        self._records = [
            replace(record, label=self.task_spec.label_map[record.phenotype["Group"]])
            for record in records
            if record.phenotype["Group"] in allowed
        ]
        if patient_ids is not None:
            selected_patients = {str(patient_id) for patient_id in patient_ids}
            self._records = [
                record for record in self._records if record.patient_id in selected_patients
            ]
        if not self._records:
            raise RuntimeError(
                f"No ADNI scans for task={self.task!r} under {self.root_path} "
                f"(kept diagnoses={list(self.task_spec.diagnoses)})"
            )
        logger.info(
            "ADNI task=%s binary=%s scans=%d labels=%s",
            self.task,
            self.task_spec.binary,
            len(self._records),
            {
                name: sum(record.label == index for record in self._records)
                for index, name in enumerate(self.class_names)
            },
        )

    @staticmethod
    def _record_metadata(record: _ScanRecord) -> Dict[str, str]:
        return dict(record.phenotype)


class ADNIClassificationDataset(_ADNIBaseDataset):
    """Whole-brain axial-slice classification.

    ``task`` selects the label space:

    * ``cn_mci_ad`` — three-class CN/MCI/AD (default)
    * ``cn_ad`` / ``cn_mci`` / ``mci_ad`` — binary Duke-style 0/1 labels
      (second diagnosis is the positive class)
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        csv_path: Optional[Union[str, Path]] = None,
        task: str = DEFAULT_ADNI_TASK,
        patient_ids: Optional[Sequence[str]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        data_timing: Optional[DataPipelineTimingProxy] = None,
    ) -> None:
        super().__init__(
            root=root,
            csv_path=csv_path,
            task=task,
            patient_ids=patient_ids,
            manifest_path=manifest_path,
            z_min=z_min,
            z_max=z_max,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            volume_cache_size=volume_cache_size,
            augment=augment,
            image_size=image_size,
        )
        self._entries: List[Tuple[_ScanRecord, int]] = []
        for record in self._records:
            start, end = _z_index_range(record.n_slices, self.z_min, self.z_max)
            self._entries.extend((record, z) for z in range(start, end))
        if not self._entries:
            raise RuntimeError(
                f"No ADNI slices found under {self.root_path} in "
                f"z-range [{self.z_min}, {self.z_max})"
            )

    def __len__(self) -> int:
        return len(self._entries)

    def get_target(self, index: int) -> int:
        return self._entries[index][0].label

    def get_patient_id(self, index: int) -> str:
        return self._entries[index][0].patient_id

    def get_image_id(self, index: int) -> str:
        return self._entries[index][0].image_id

    def get_phenotype_raw(self, index: int) -> Dict[str, str]:
        return self._record_metadata(self._entries[index][0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        assert self._volume_cache is not None
        volume = self._volume_cache.get(record.volume_path)
        image: Any = _slice_to_pil(volume, z)
        target: Any = record.label

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return image, target


class ADNIMultiSliceDataset(_ADNIBaseDataset):
    """Scan-level classification using an interpolated stack of axial slices.

    ``task`` selects the label space (see :class:`ADNIClassificationDataset`).
    Binary tasks mirror Duke: labels are ``0``/``1`` with a single-logit BCE head.
    Each volume is resampled to exactly ``n_slices`` slices with trilinear
    interpolation. When enabled, MONAI 3-D transforms are applied once to the
    complete volume rather than independently to each slice.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        n_slices: int = 8,
        csv_path: Optional[Union[str, Path]] = None,
        task: str = DEFAULT_ADNI_TASK,
        patient_ids: Optional[Sequence[str]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = True,
        image_size: int = 224,
        return_imagenet_tensors: bool = True,
        three_d_encoder: bool = False,
        data_timing: Optional[DataPipelineTimingProxy] = None,
    ) -> None:
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        self.n_slices = int(n_slices)
        self.image_size = int(image_size)
        # ``return_imagenet_tensors`` is retained for direct callers of the
        # earlier NeuroVFM integration.  ``three_d_encoder`` is the public,
        # consistent option shared by all multi-slice datasets.
        self.return_imagenet_tensors = bool(return_imagenet_tensors)
        self.three_d_encoder = bool(three_d_encoder)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if transforms is None and transform is None and augment:
            transform = build_adni_volume_transform(augment=True)
        super().__init__(
            root=root,
            csv_path=csv_path,
            task=task,
            patient_ids=patient_ids,
            manifest_path=manifest_path,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            # Each item consumes its entire volume once, so a per-worker cache
            # only holds large arrays without useful reuse.
            volume_cache_size=None,
            augment=augment,
            image_size=image_size,
            build_default_transform=False,
            data_timing=data_timing,
        )
        self._entries = list(self._records)

    def __len__(self) -> int:
        return len(self._entries)

    def get_target(self, index: int) -> int:
        return self._entries[index].label

    def get_patient_id(self, index: int) -> str:
        return self._entries[index].patient_id

    def get_image_id(self, index: int) -> str:
        return self._entries[index].image_id

    def get_phenotype_raw(self, index: int) -> Dict[str, str]:
        return self._record_metadata(self._entries[index])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record = self._entries[index]
        read_start = time.perf_counter()
        volume = _load_canonical_volume(record.volume_path)
        read_seconds = time.perf_counter() - read_start
        preprocessing_start = time.perf_counter()
        volume_t = _resample_volume(
            _scale_volume_to_unit_interval(volume), self.n_slices, self.image_size
        ).unsqueeze(0)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        target: Any = record.label

        augmentation_start = time.perf_counter()
        if self.transforms is not None:
            volume_t, target = self.transforms(volume_t, target)
        else:
            if self.transform is not None:
                volume_t = self.transform(volume_t)
            if self.target_transform is not None:
                target = self.target_transform(target)

        augmentation_seconds = (
            time.perf_counter() - augmentation_start
            if self.transform is not None or self.transforms is not None
            else 0.0
        )
        conversion_start = time.perf_counter()
        volume_t = torch.as_tensor(volume_t)
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)
        if volume_t.ndim != 4 or volume_t.shape[0] != 1:
            raise ValueError(
                "ADNI multi-slice transforms must return [1, depth, height, width], "
                f"got shape {tuple(volume_t.shape)}"
            )
        if getattr(self, "three_d_encoder", False):
            # Dataset boundary for volume encoders: [D, 1, H, W].  Values
            # retain the single robust volume scaling applied before resampling.
            images = volume_t.permute(1, 0, 2, 3)
        elif getattr(self, "return_imagenet_tensors", True):
            images = _volume_to_imagenet_tensors(volume_t.squeeze(0))
        else:
            images = volume_t
        preprocessing_seconds += time.perf_counter() - conversion_start
        if self.data_timing is not None:
            self.data_timing.add(
                volume_read_s=read_seconds,
                volume_preprocessing_s=preprocessing_seconds,
                volume_augmentation_s=augmentation_seconds,
            )
        return images, target


def _encode_phenotypes(
    phenotype: Dict[str, str],
    columns: Sequence[str],
) -> np.ndarray:
    values: List[float] = []
    for column in columns:
        raw_value = phenotype.get(column, "").strip()
        if column == "Group":
            value = float(DIAGNOSIS_TO_LABEL.get(raw_value.upper(), PHENOTYPE_SENTINEL))
        elif column == "Sex":
            value = {"F": 0.0, "M": 1.0}.get(raw_value.upper(), PHENOTYPE_SENTINEL)
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = PHENOTYPE_SENTINEL
        values.append(value)
    return np.asarray(values, dtype=np.float32)


class ADNIPairedSliceDataset(ADNIClassificationDataset):
    """Paired slices from the same ADNI volume.

    Each item returns ``(slice_1, slice_2)``. Both slices are sampled from the
    same volume, and a DataLoader collates them into ``(slices1, slices2)``
    with each tensor shaped ``[B, 3, H, W]``.

    The phenotype helpers are retained for callers that need metadata, but the
    paired SSL path does not return a target from ``__getitem__``.
    ``patient_ids``, when provided, restricts the dataset to those patients.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        max_distance: int = 3,
        patient_ids: Optional[Sequence[str]] = None,
        phenotype_columns: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        self.max_distance = int(max_distance)
        self.selected_phenotype_columns = list(
            phenotype_columns or DEFAULT_PHENOTYPE_COLUMNS
        )
        self._seed = seed
        super().__init__(root=root, patient_ids=patient_ids, **kwargs)
        unknown_columns = set(self.selected_phenotype_columns).difference(
            self.phenotype_columns
        )
        if unknown_columns:
            raise ValueError(
                f"Unknown phenotype columns: {sorted(unknown_columns)}; "
                f"available columns are {self.phenotype_columns}"
            )

    def get_target(self, index: int) -> np.ndarray:
        return _encode_phenotypes(
            self.get_phenotype_raw(index),
            self.selected_phenotype_columns,
        )

    def _rng_for_index(self, index: int) -> np.random.RandomState:
        if self._seed is None:
            return np.random.RandomState()
        return np.random.RandomState(self._seed + int(index))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        start, end = _z_index_range(record.n_slices, self.z_min, self.z_max)
        low = max(start, z - self.max_distance)
        high = min(end - 1, z + self.max_distance)
        candidates = [partner for partner in range(low, high + 1) if partner != z]
        partner_z = (
            int(self._rng_for_index(index).choice(candidates)) if candidates else z
        )

        volume = self._volume_cache.get(record.volume_path)
        image: Any = (
            _slice_to_pil(volume, z),
            _slice_to_pil(volume, partner_z),
        )
        if self.transform is not None:
            image = _apply_shared_pair_transform(self.transform, image)
        elif self.transforms is not None:
            transformed = self.transforms(image, None)
            image = transformed[0] if isinstance(transformed, tuple) else transformed
        return image


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
