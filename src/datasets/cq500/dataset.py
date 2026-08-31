"""Paired-slice SSL and volume-level classification datasets for CQ500 CT."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

from .convert import DEFAULT_ROOT, META_NAME, VOLUME_NAME

logger = logging.getLogger(__name__)

DEFAULT_LABELS_CSV = DEFAULT_ROOT / "reads.csv"
ICH_SUBTYPES: Tuple[str, ...] = ("IPH", "IVH", "SDH", "EDH", "SAH")
ICH_SUBTYPE_COLUMNS: Tuple[str, ...] = tuple(f"R1:{name}" for name in ICH_SUBTYPES)
_REQUIRED_LABEL_COLUMNS = ("name", "R1:ICH", *ICH_SUBTYPE_COLUMNS)
CQ500_TASK_CHOICES = ("ich", "subtype")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CQ500TaskSpec:
    """Label definition for a CQ500 classification task."""

    name: str
    class_names: Tuple[str, ...]
    binary: bool
    multi_label: bool

    @property
    def num_logits(self) -> int:
        """Output width for a classifier head."""
        return 1 if self.binary else len(self.class_names)


_TASK_SPECS = {
    "ich": CQ500TaskSpec("ich", ("no_ich", "ich"), binary=True, multi_label=False),
    # CQ500 has co-occurring subtypes, so this is a five-logit multi-label task
    # (rather than lossy single-class assignment).
    "subtype": CQ500TaskSpec("subtype", ICH_SUBTYPES, binary=False, multi_label=True),
}


def resolve_cq500_task(task: str) -> CQ500TaskSpec:
    """Resolve ``ich`` or ``subtype`` task names (with useful aliases)."""
    normalized = str(task).strip().lower().replace("-", "_")
    aliases = {"binary": "ich", "binary_ich": "ich", "ich_subtype": "subtype"}
    normalized = aliases.get(normalized, normalized)
    spec = _TASK_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"Unknown CQ500 task={task!r}; choose from {CQ500_TASK_CHOICES}")
    return spec


def _normalize_patient_id(value: Any) -> str:
    value = str(value).strip()
    if value.startswith("CQ500-CT-"):
        return "CQ500CT" + value.removeprefix("CQ500-CT-")
    if value.startswith("CQ500CT"):
        return value
    raise ValueError(f"Invalid CQ500 patient ID {value!r}")


def _binary_value(value: Any, *, column: str, patient_id: str) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column}={value!r} for {patient_id}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"Expected {column} to be 0 or 1 for {patient_id}, got {value!r}")
    return parsed


def read_cq500_labels(csv_path: Union[str, Path] = DEFAULT_LABELS_CSV) -> Dict[str, Dict[str, Any]]:
    """Read R1 CQ500 labels keyed by filesystem-style patient ID."""
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing CQ500 label CSV: {path}")
    labels: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = set(_REQUIRED_LABEL_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"CQ500 labels missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            patient_id = _normalize_patient_id(row["name"])
            if patient_id in labels:
                raise ValueError(f"Duplicate CQ500 label for {patient_id} at line {line_number}")
            ich = _binary_value(row["R1:ICH"], column="R1:ICH", patient_id=patient_id)
            subtype = np.asarray(
                [
                    _binary_value(row[column], column=column, patient_id=patient_id)
                    for column in ICH_SUBTYPE_COLUMNS
                ],
                dtype=np.float32,
            )
            labels[patient_id] = {"ich": ich, "subtype": subtype, "raw": dict(row)}
    return labels


def _apply_shared_pair_transform(transform: Callable, images: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """Apply one stochastic transform realization identically to both slices."""
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    first = transform(images[0])
    advanced_torch = torch.get_rng_state()
    advanced_numpy = np.random.get_state()
    advanced_python = random.getstate()
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    second = transform(images[1])
    torch.set_rng_state(advanced_torch)
    np.random.set_state(advanced_numpy)
    random.setstate(advanced_python)
    return first, second


class _VolumeCache:
    """Process-local LRU cache of canonical, mmap-backed NIfTI arrays."""

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

        image = nib.as_closest_canonical(nib.load(key, mmap="r"))
        volume = np.asanyarray(image.dataobj)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3-D NIfTI volume at {path}, got {volume.shape}")
        self._cache[key] = volume
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return volume


@dataclass(frozen=True)
class _VolumeRecord:
    patient_id: str
    scan_name: str
    volume_path: Path
    n_slices: int
    slice_spacing_mm: float
    voxel_spacing_mm: Tuple[float, float, float]


def _canonical_metadata(volume_path: Path) -> Tuple[int, Tuple[float, float, float]]:
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(volume_path), mmap="r"))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3-D NIfTI at {volume_path}, got {image.shape}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return int(image.shape[-1]), spacing


def _discover_volume_records(root: Path) -> List[_VolumeRecord]:
    records: List[_VolumeRecord] = []
    for volume_path in sorted(root.glob(f"CQ500CT*/*/{VOLUME_NAME}")):
        relative = volume_path.relative_to(root)
        if len(relative.parts) != 3:
            continue
        patient_id, scan_name, _ = relative.parts
        n_slices, spacing = _canonical_metadata(volume_path)
        meta_path = volume_path.with_name(META_NAME)
        # The sidecar is preserved as the conversion record, but calculate the
        # canonical spacing above because the dataset indexes canonical z slices.
        if meta_path.is_file():
            with meta_path.open() as handle:
                metadata = json.load(handle)
            if metadata.get("patient_id") not in (None, patient_id):
                raise ValueError(f"Metadata patient mismatch in {meta_path}")
        records.append(
            _VolumeRecord(
                patient_id=patient_id,
                scan_name=scan_name,
                volume_path=volume_path,
                n_slices=n_slices,
                slice_spacing_mm=float(spacing[2]),
                voxel_spacing_mm=spacing,
            )
        )
    if not records:
        raise RuntimeError(f"No {VOLUME_NAME} files found under {root}; run CQ500 conversion first.")
    return records


def _slice_to_pil(volume: np.ndarray, z: int) -> Image.Image:
    image_slice = np.asarray(volume[..., z], dtype=np.float32)
    finite = image_slice[np.isfinite(image_slice)]
    if finite.size == 0:
        raise ValueError(f"CQ500 slice {z} contains no finite values")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    scaled = np.nan_to_num(np.clip((image_slice - low) / (high - low), 0.0, 1.0))
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8), mode="L").convert("RGB")


def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("CQ500 volume contains no finite values")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        high = low + 1.0
    normalized = np.nan_to_num(np.clip(volume, low, high), nan=low, posinf=high, neginf=low)
    return ((normalized - float(normalized.mean())) / max(float(normalized.std()), 1e-6)).astype(np.float32)


def _resample_volume(volume: np.ndarray, n_slices: int, image_size: int) -> torch.Tensor:
    if n_slices <= 0 or image_size <= 0:
        raise ValueError("n_slices and image_size must be positive")
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(n_slices, volume.shape[0], volume.shape[1]), mode="trilinear", align_corners=False)
    tensor = F.interpolate(tensor.squeeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def _volume_to_rgb(volume: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``[D,H,W]`` volume to ImageNet-normalized RGB slices."""
    low = volume.amin(dim=(1, 2), keepdim=True)
    high = volume.amax(dim=(1, 2), keepdim=True)
    scaled = torch.where(high > low, (volume - low) / (high - low), torch.zeros_like(volume))
    images = scaled.clamp(0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def build_cq500_transform(image_size: int = 224, *, augment: bool = True) -> tv_transforms.Compose:
    """Default paired-slice CT transform."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    operations: List[Callable] = []
    if augment:
        operations.extend(
            [
                tv_transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                tv_transforms.RandomRotation(15),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    else:
        operations.append(tv_transforms.Resize((image_size, image_size)))
    operations.extend([tv_transforms.ToTensor(), tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tv_transforms.Compose(operations)


def build_cq500_volume_transform(*, augment: bool = True) -> Optional[Callable]:
    """Build a MONAI transform applied once to an entire ``[1,D,H,W]`` volume."""
    if not augment:
        return None
    try:
        from monai.transforms import Compose, RandAdjustContrastd, RandAffined, RandFlipd, RandGaussianNoised
    except ImportError as exc:
        raise ImportError("MONAI is required for CQ500 volume augmentation; set augment=False or install monai") from exc
    transform = Compose(
        [
            RandAffined(keys=("image",), rotate_range=(0.1, 0.1, 0.1), translate_range=(5, 5, 5), scale_range=(0.1, 0.1, 0.1), prob=0.5, padding_mode="border", mode="trilinear"),
            RandFlipd(keys=("image",), spatial_axis=[2], prob=0.5),
            RandGaussianNoised(keys=("image",), prob=0.2, std=0.05),
            RandAdjustContrastd(keys=("image",), prob=0.2, gamma=(0.7, 1.3)),
        ]
    )
    return lambda volume: transform({"image": volume})["image"]


class _CQ500BaseDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        csv_path: Optional[Union[str, Path]] = None,
        patient_ids: Optional[Sequence[str]] = None,
        load_labels: bool = True,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        super().__init__(str(root_path), transforms=transforms, transform=transform, target_transform=target_transform)
        self.root_path = root_path
        self.labels = read_cq500_labels(csv_path or DEFAULT_LABELS_CSV) if load_labels else {}
        records = _discover_volume_records(root_path)
        selected = {_normalize_patient_id(value) for value in patient_ids} if patient_ids is not None else None
        self._records = [
            record for record in records
            if (not load_labels or record.patient_id in self.labels)
            and (selected is None or record.patient_id in selected)
        ]
        if not self._records:
            raise RuntimeError(f"No labelled CQ500 volumes under {root_path}")
        self._volume_cache = _VolumeCache(volume_cache_size)

    def get_patient_id(self, index: int) -> str:
        return self._records[index].patient_id

    def get_volume_metadata(self, index: int) -> Dict[str, Any]:
        """Return physical spacing and provenance for the volume at ``index``."""
        record = self._records[index]
        return {
            "patient_id": record.patient_id,
            "scan_name": record.scan_name,
            "volume_path": str(record.volume_path),
            "n_slices": record.n_slices,
            "slice_spacing_mm": record.slice_spacing_mm,
            "voxel_spacing_mm": record.voxel_spacing_mm,
        }

    def get_slice_spacing_mm(self, index: int) -> float:
        """Return the physical z-axis spacing for the indexed volume."""
        return float(self.get_volume_metadata(index)["slice_spacing_mm"])


class CQ500PairedSliceDataset(_CQ500BaseDataset):
    """Unlabelled paired axial slices with shared stochastic augmentation.

    ``get_volume_metadata`` exposes the original volume's slice spacing for any
    entry.  ``__getitem__`` deliberately returns only ``(slice_a, slice_b)``.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        patient_ids: Optional[Sequence[str]] = None,
        max_distance: int = 3,
        z_min: float = 0.0,
        z_max: float = 1.0,
        seed: Optional[int] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if max_distance < 0 or not 0 <= z_min <= z_max <= 1:
            raise ValueError("Require max_distance >= 0 and 0 <= z_min <= z_max <= 1")
        if transforms is None and transform is None:
            transform = build_cq500_transform(image_size=image_size, augment=augment)
        super().__init__(root, csv_path=csv_path, patient_ids=patient_ids, load_labels=False, transforms=transforms, transform=transform, target_transform=target_transform, volume_cache_size=volume_cache_size)
        self.max_distance, self.z_min, self.z_max, self._seed = int(max_distance), float(z_min), float(z_max), seed
        self._entries: List[Tuple[_VolumeRecord, int]] = []
        for record in self._records:
            start, end = int(np.floor(record.n_slices * z_min)), int(np.ceil(record.n_slices * z_max))
            self._entries.extend((record, z) for z in range(start, end))
        if not self._entries:
            raise RuntimeError("No CQ500 slices in the requested z range")

    def __len__(self) -> int:
        return len(self._entries)

    def get_patient_id(self, index: int) -> str:
        return self._entries[index][0].patient_id

    def get_volume_metadata(self, index: int) -> Dict[str, Any]:
        record = self._entries[index][0]
        return {
            "patient_id": record.patient_id, "scan_name": record.scan_name,
            "volume_path": str(record.volume_path), "n_slices": record.n_slices,
            "slice_spacing_mm": record.slice_spacing_mm, "voxel_spacing_mm": record.voxel_spacing_mm,
        }

    def get_slice_spacing_mm(self, index: int) -> float:
        return float(self._entries[index][0].slice_spacing_mm)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        lo, hi = max(0, z - self.max_distance), min(record.n_slices - 1, z + self.max_distance)
        choices = [candidate for candidate in range(lo, hi + 1) if candidate != z]
        rng = np.random.RandomState(None if self._seed is None else self._seed + index)
        partner_z = int(rng.choice(choices)) if choices else z
        volume = self._volume_cache.get(record.volume_path)
        images: Any = (_slice_to_pil(volume, z), _slice_to_pil(volume, partner_z))
        if self.transform is not None:
            return _apply_shared_pair_transform(self.transform, images)
        if self.transforms is not None:
            transformed = self.transforms(images, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return images


class CQ500SliceDataset(_CQ500BaseDataset):
    """Single axial slices for supervised CQ500 classification.

    Each item returns ``(image, target)``.  ``task='ich'`` produces an integer
    binary target; ``task='subtype'`` retains ICH-positive volumes only and
    returns the five-element multi-hot subtype vector.  Physical spacing is
    available through :meth:`get_volume_metadata` and
    :meth:`get_slice_spacing_mm`.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        task: str = "ich",
        patient_ids: Optional[Sequence[str]] = None,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not 0.0 <= z_min <= z_max <= 1.0:
            raise ValueError(f"Require 0 <= z_min <= z_max <= 1, got {z_min}, {z_max}")
        if transforms is None and transform is None:
            transform = build_cq500_transform(image_size=image_size, augment=augment)
        self.task_spec = resolve_cq500_task(task)
        self.task, self.class_names = self.task_spec.name, self.task_spec.class_names
        self.z_min, self.z_max = float(z_min), float(z_max)
        super().__init__(
            root,
            csv_path=csv_path,
            patient_ids=patient_ids,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            volume_cache_size=volume_cache_size,
        )
        if self.task == "subtype":
            self._records = [
                record for record in self._records if self.labels[record.patient_id]["ich"] == 1
            ]
        self._entries: List[Tuple[_VolumeRecord, int]] = []
        for record in self._records:
            start = int(np.floor(record.n_slices * self.z_min))
            end = int(np.ceil(record.n_slices * self.z_max))
            self._entries.extend((record, z) for z in range(start, end))
        if not self._entries:
            raise RuntimeError(f"No CQ500 slices remain for task={self.task!r}")

    def __len__(self) -> int:
        return len(self._entries)

    def get_patient_id(self, index: int) -> str:
        return self._entries[index][0].patient_id

    def get_volume_metadata(self, index: int) -> Dict[str, Any]:
        record = self._entries[index][0]
        return {
            "patient_id": record.patient_id,
            "scan_name": record.scan_name,
            "volume_path": str(record.volume_path),
            "n_slices": record.n_slices,
            "slice_spacing_mm": record.slice_spacing_mm,
            "voxel_spacing_mm": record.voxel_spacing_mm,
        }

    def get_slice_spacing_mm(self, index: int) -> float:
        return float(self._entries[index][0].slice_spacing_mm)

    def get_target(self, index: int) -> Union[int, np.ndarray]:
        label = self.labels[self._entries[index][0].patient_id]
        return int(label["ich"]) if self.task == "ich" else label["subtype"].copy()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        image: Any = _slice_to_pil(self._volume_cache.get(record.volume_path), z)
        target: Any = self.get_target(index)
        if self.transforms is not None:
            return self.transforms(image, target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class CQ500MultiSliceDataset(_CQ500BaseDataset):
    """Whole-volume CQ500 classification with optional 3-D augmentation.

    ``task='ich'`` returns an integer binary target from ``R1:ICH``.  For
    ``task='subtype'`` only ICH-positive patients are retained and the target is
    a five-element float32 multi-hot vector in ``ICH_SUBTYPES`` order.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        task: str = "ich",
        patient_ids: Optional[Sequence[str]] = None,
        n_slices: int = 32,
        image_size: int = 224,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if n_slices <= 0 or image_size <= 0:
            raise ValueError("n_slices and image_size must be positive")
        self.task_spec = resolve_cq500_task(task)
        self.task, self.class_names = self.task_spec.name, self.task_spec.class_names
        self.n_slices, self.image_size = int(n_slices), int(image_size)
        if transforms is None and transform is None and augment:
            transform = build_cq500_volume_transform(augment=True)
        super().__init__(root, csv_path=csv_path, patient_ids=patient_ids, transforms=transforms, transform=transform, target_transform=target_transform, volume_cache_size=volume_cache_size)
        if self.task == "subtype":
            self._records = [record for record in self._records if self.labels[record.patient_id]["ich"] == 1]
        if not self._records:
            raise RuntimeError(f"No CQ500 volumes remain for task={self.task!r}")

    def __len__(self) -> int:
        return len(self._records)

    def get_target(self, index: int) -> Union[int, np.ndarray]:
        label = self.labels[self._records[index].patient_id]
        return int(label["ich"]) if self.task == "ich" else label["subtype"].copy()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        record = self._records[index]
        volume = self._volume_cache.get(record.volume_path)
        image = _resample_volume(_zscore_normalize(volume), self.n_slices, self.image_size).unsqueeze(0)
        target: Any = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        image = torch.as_tensor(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"CQ500 volume transform must return [1,D,H,W], got {tuple(image.shape)}")
        return _volume_to_rgb(image.squeeze(0)), target


__all__ = [
    "CQ500MultiSliceDataset", "CQ500PairedSliceDataset", "CQ500SliceDataset", "CQ500TaskSpec",
    "CQ500_TASK_CHOICES", "DEFAULT_LABELS_CSV", "DEFAULT_ROOT", "ICH_SUBTYPES",
    "build_cq500_transform", "build_cq500_volume_transform", "read_cq500_labels",
    "resolve_cq500_task",
]
