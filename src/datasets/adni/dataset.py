"""PyTorch datasets for the ADNI1 1.5T MRI collection.

The source dataset is consumed in place: NIfTI files are matched to the ADNI
CSV by the image ID stored in each file's direct parent directory.  No
conversion or segmentation products are required.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPOSITORY_ROOT / "data" / "ADNI"
DEFAULT_CSV_NAME = "ADNI1_Complete_3Yr_1.5T_7_21_2026.csv"

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
    cuda_initial = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    first = transform(images[0])
    torch_advanced = torch.get_rng_state()
    numpy_advanced = np.random.get_state()
    python_advanced = random.getstate()
    cuda_advanced = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.set_rng_state(torch_initial)
    np.random.set_state(numpy_initial)
    random.setstate(python_initial)
    if cuda_initial is not None:
        torch.cuda.set_rng_state_all(cuda_initial)
    second = transform(images[1])

    torch.set_rng_state(torch_advanced)
    np.random.set_state(numpy_advanced)
    random.setstate(python_advanced)
    if cuda_advanced is not None:
        torch.cuda.set_rng_state_all(cuda_advanced)
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

        import nibabel as nib

        image = nib.load(key, mmap="r")
        canonical = nib.as_closest_canonical(image)
        volume = np.asanyarray(canonical.dataobj)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume at {path}, got shape {volume.shape}")

        self._cache[key] = volume
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
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


def _discover_volumes(root: Path) -> Dict[str, Path]:
    volumes: Dict[str, Path] = {}
    for path in sorted((*root.rglob("*.nii"), *root.rglob("*.nii.gz"))):
        image_id = _normalize_image_id(path.parent.name)
        previous = volumes.get(image_id)
        if previous is not None:
            raise ValueError(
                f"Multiple NIfTI files found for image ID {image_id}: {previous} and {path}"
            )
        volumes[image_id] = path
    if not volumes:
        raise RuntimeError(f"No .nii or .nii.gz files found under {root}")
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


def _build_scan_index(root: Path, csv_path: Path) -> Tuple[List[_ScanRecord], List[str]]:
    volumes = _discover_volumes(root)
    columns, metadata = _read_metadata(csv_path)

    file_ids = set(volumes)
    metadata_ids = set(metadata)
    missing_files = sorted(metadata_ids - file_ids)
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
                n_slices=_canonical_depth(path),
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


SLICE_SAMPLING_CHOICES = ("random", "even")


def _evenly_spaced_slice_indices(start: int, end: int, n_slices: int) -> Tuple[int, ...]:
    """Return ``n_slices`` deterministic indices spanning ``[start, end)``."""
    eligible_slices = end - start
    if eligible_slices <= 0:
        raise ValueError(f"Empty z-range [{start}, {end})")
    if n_slices <= 0:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    if n_slices > eligible_slices:
        raise ValueError(
            f"Cannot place {n_slices} evenly spaced slices in "
            f"{eligible_slices} eligible slices"
        )
    if n_slices == 1:
        return (start + eligible_slices // 2,)
    span = eligible_slices - 1
    return tuple(
        start + int(round(index * span / (n_slices - 1))) for index in range(n_slices)
    )


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


class _ADNIBaseDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        csv_path: Optional[Union[str, Path]] = None,
        task: str = DEFAULT_ADNI_TASK,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        metadata_path = (
            Path(csv_path).expanduser().resolve()
            if csv_path is not None
            else root_path / DEFAULT_CSV_NAME
        )
        if transforms is None and transform is None:
            transform = build_adni_transform(image_size=image_size, augment=augment)
        super().__init__(
            str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root_path
        self.csv_path = metadata_path
        self.task_spec = resolve_adni_task(task)
        self.task = self.task_spec.name
        self.class_names = self.task_spec.class_names
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        _z_index_range(1, self.z_min, self.z_max)
        self._volume_cache = _VolumeCache(volume_cache_size)
        records, self.phenotype_columns = _build_scan_index(root_path, metadata_path)
        allowed = set(self.task_spec.diagnoses)
        self._records = [
            replace(record, label=self.task_spec.label_map[record.phenotype["Group"]])
            for record in records
            if record.phenotype["Group"] in allowed
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
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__(
            root=root,
            csv_path=csv_path,
            task=task,
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
    """Scan-level classification using an ordered stack of axial slices.

    ``task`` selects the label space (see :class:`ADNIClassificationDataset`).
    Binary tasks mirror Duke: labels are ``0``/``1`` with a single-logit BCE head.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        n_slices: int = 8,
        csv_path: Optional[Union[str, Path]] = None,
        task: str = DEFAULT_ADNI_TASK,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        minimum_z_index_distance: Optional[int] = None,
        slice_sampling: str = "random",
    ) -> None:
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if minimum_z_index_distance is not None and minimum_z_index_distance < 0:
            raise ValueError(
                "minimum_z_index_distance must be non-negative or None, "
                f"got {minimum_z_index_distance}"
            )
        if slice_sampling not in SLICE_SAMPLING_CHOICES:
            raise ValueError(
                f"slice_sampling must be one of {SLICE_SAMPLING_CHOICES}, "
                f"got {slice_sampling!r}"
            )
        self.n_slices = int(n_slices)
        self.minimum_z_index_distance = minimum_z_index_distance
        self.slice_sampling = slice_sampling
        super().__init__(
            root=root,
            csv_path=csv_path,
            task=task,
            z_min=z_min,
            z_max=z_max,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            volume_cache_size=volume_cache_size,
            augment=augment,
            image_size=image_size,
        )
        self._entries = list(self._records)
        for record in self._entries:
            self._validate_slice_capacity(record)

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

    def _sampling_parameters(self, record: _ScanRecord) -> Tuple[int, int, int]:
        start, end = _z_index_range(record.n_slices, self.z_min, self.z_max)
        eligible_slices = end - start
        distance = self.minimum_z_index_distance
        if distance is None:
            distance = max(eligible_slices // self.n_slices - 1, 0)
        return start, end, int(distance)

    def _validate_slice_capacity(self, record: _ScanRecord) -> None:
        start, end, distance = self._sampling_parameters(record)
        eligible_slices = end - start
        if self.slice_sampling == "even":
            if self.n_slices > eligible_slices:
                raise ValueError(
                    f"Cannot place {self.n_slices} evenly spaced slices in "
                    f"{eligible_slices} eligible slices for image ID {record.image_id}"
                )
            return
        effective_distance = max(distance, 1)
        required_span = 1 + (self.n_slices - 1) * effective_distance
        if required_span > eligible_slices:
            raise ValueError(
                f"Cannot sample {self.n_slices} slices with minimum z-index "
                f"distance {distance} from {eligible_slices} eligible slices "
                f"for image ID {record.image_id}"
            )

    def get_slice_indices(self, index: int) -> Tuple[int, ...]:
        record = self._entries[index]
        start, end, distance = self._sampling_parameters(record)
        if self.slice_sampling == "even":
            try:
                return _evenly_spaced_slice_indices(start, end, self.n_slices)
            except ValueError as exc:
                raise ValueError(f"{exc} for image ID {record.image_id}") from exc

        eligible_slices = end - start
        effective_distance = max(distance, 1)
        compressed_size = eligible_slices - (
            effective_distance - 1
        ) * (self.n_slices - 1)
        compressed = torch.randperm(compressed_size)[: self.n_slices]
        compressed, _ = torch.sort(compressed)
        offsets = torch.arange(self.n_slices) * (effective_distance - 1)
        return tuple(int(z) for z in (compressed + offsets + start).tolist())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record = self._entries[index]
        volume = self._volume_cache.get(record.volume_path)
        images: List[Any] = [
            _slice_to_pil(volume, z) for z in self.get_slice_indices(index)
        ]
        target: Any = record.label

        if self.transform is not None:
            initial_rng_state = torch.get_rng_state()
            images[0] = self.transform(images[0])
            advanced_rng_state = torch.get_rng_state()
            for image_index in range(1, len(images)):
                torch.set_rng_state(initial_rng_state)
                images[image_index] = self.transform(images[image_index])
            torch.set_rng_state(advanced_rng_state)
            if self.target_transform is not None:
                target = self.target_transform(target)
        elif self.transforms is not None:
            transformed = [self.transforms(image, target) for image in images]
            images = [pair[0] for pair in transformed]
            target = transformed[0][1]
        elif self.target_transform is not None:
            target = self.target_transform(target)

        if images and all(torch.is_tensor(image) for image in images):
            return torch.stack(images, dim=0), target
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
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        max_distance: int = 3,
        n_patients: Optional[int] = None,
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
        super().__init__(root=root, **kwargs)
        patient_ids = list(dict.fromkeys(record.patient_id for record, _ in self._entries))
        if n_patients is None:
            n_patients = len(patient_ids)
        if n_patients <= 0:
            raise ValueError(f"n_patients must be positive, got {n_patients}")
        if n_patients > len(patient_ids):
            raise ValueError(
                f"Requested n_patients={n_patients}, but only {len(patient_ids)} patients are available"
            )
        selected_patients = set(patient_ids[:n_patients])
        self._entries = [
            (record, z)
            for record, z in self._entries
            if record.patient_id in selected_patients
        ]
        self.n_patients = n_patients
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
    "DEFAULT_PHENOTYPE_COLUMNS",
    "DEFAULT_ROOT",
    "DIAGNOSIS_TO_LABEL",
    "LABEL_TO_DIAGNOSIS",
    "PHENOTYPE_SENTINEL",
    "build_adni_transform",
    "resolve_adni_task",
]
