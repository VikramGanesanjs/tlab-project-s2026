"""Slice and volume datasets for the supplied BreastDM DCE-MRI splits.

The source ``img17Se`` directory is authoritative: it contains 17-series
``uint8`` NumPy arrays under ``<split>/<class>/<patient>/<scan>.npy``.  The
provided split directories are consumed directly; no patient-level split is
created by these datasets.
"""

from __future__ import annotations

import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPOSITORY_ROOT / "data" / "breastdm" / "cls"
IMAGE_DIRECTORY = "img17Se"
SPLIT_CHOICES = ("train", "val", "test")
BREASTDM_CLASS_NAMES = ("Benign", "Malignant")
LABEL_TO_INDEX = {name: index for index, name in enumerate(BREASTDM_CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _apply_shared_pair_transform(
    transform: Callable, images: Tuple[Any, Any]
) -> Tuple[Any, Any]:
    """Apply identical stochastic augmentation parameters to a slice pair."""
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
class _VolumeRecord:
    split: str
    class_name: str
    label: int
    patient_id: str
    scan_id: str
    volume_path: Path
    n_slices: int


class _VolumeCache:
    """Small process-local LRU cache of read-only NumPy arrays."""

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
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3-D NumPy array at {path}, got {volume.shape}")
        self._cache[key] = volume
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return volume


def _image_root(root: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolve the classification root and its mandatory ``img17Se`` child."""
    supplied = Path(root).expanduser().resolve()
    image_root = supplied if supplied.name == IMAGE_DIRECTORY else supplied / IMAGE_DIRECTORY
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"BreastDM {IMAGE_DIRECTORY} directory does not exist: {image_root}"
        )
    return image_root.parent, image_root


def _discover_records(image_root: Path, split: str) -> List[_VolumeRecord]:
    if split not in SPLIT_CHOICES:
        raise ValueError(f"Unknown split={split!r}; choose from {SPLIT_CHOICES}")
    split_root = image_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"BreastDM split directory does not exist: {split_root}")

    records: List[_VolumeRecord] = []
    for class_name, label in LABEL_TO_INDEX.items():
        class_root = split_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(
                f"BreastDM class directory does not exist: {class_root}"
            )
        for volume_path in sorted(class_root.glob("*/*.npy")):
            patient_id = volume_path.parent.name
            array = np.load(volume_path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3:
                raise ValueError(
                    f"Expected a 3-D NumPy array at {volume_path}, got {array.shape}"
                )
            if array.shape[-1] != 17:
                raise ValueError(
                    f"Expected 17 series in {volume_path}, got shape {array.shape}"
                )
            records.append(
                _VolumeRecord(
                    split=split,
                    class_name=class_name,
                    label=label,
                    patient_id=patient_id,
                    scan_id=volume_path.stem,
                    volume_path=volume_path,
                    n_slices=int(array.shape[-1]),
                )
            )
    if not records:
        raise RuntimeError(f"No BreastDM .npy volumes found in {split_root}")
    return records


def _slice_to_pil(volume: np.ndarray, z: int) -> Image.Image:
    """Convert one native-resolution BreastDM series image to RGB PIL."""
    image_slice = np.asarray(volume[..., z], dtype=np.float32)
    finite = image_slice[np.isfinite(image_slice)]
    if finite.size == 0:
        raise ValueError(f"BreastDM slice {z} contains no finite values")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    image_slice = np.nan_to_num(
        np.clip((image_slice - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return Image.fromarray(
        np.rint(image_slice * 255.0).astype(np.uint8), mode="L"
    ).convert("RGB")


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Robustly normalize a complete BreastDM volume before 3-D augmentation."""
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("BreastDM volume contains no finite values")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return np.nan_to_num(
        np.clip((volume - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def _resample_volume(volume: np.ndarray, n_slices: int, image_size: int) -> torch.Tensor:
    """Return ``[depth, image_size, image_size]`` from a ``[H, W, depth]`` array."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")
    if n_slices <= 0 or image_size <= 0:
        raise ValueError("n_slices and image_size must be positive")
    height, width, _ = volume.shape
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).permute(2, 0, 1)
    tensor = F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=(int(n_slices), height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    return F.interpolate(
        tensor,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _volume_to_imagenet_tensors(volume: torch.Tensor) -> torch.Tensor:
    """Convert ``[D, H, W]`` grayscale slices into normalized RGB tensors."""
    slices = torch.as_tensor(volume, dtype=torch.float32)
    if slices.ndim != 3:
        raise ValueError(f"Expected [D, H, W], got {tuple(slices.shape)}")
    finite = torch.isfinite(slices)
    if not bool(finite.flatten(1).any(dim=1).all()):
        raise ValueError("BreastDM slice contains no finite values")
    low = torch.where(finite, slices, torch.full_like(slices, float("inf"))).amin(
        dim=(1, 2), keepdim=True
    )
    high = torch.where(
        finite, slices, torch.full_like(slices, float("-inf"))
    ).amax(dim=(1, 2), keepdim=True)
    scaled = torch.where(high > low, (slices - low) / (high - low), torch.zeros_like(slices))
    images = torch.nan_to_num(scaled.clamp(0.0, 1.0), nan=0.0).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def build_breastdm_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
) -> tv_transforms.Compose:
    """Build the 2-D preprocessing used by BreastDM slice datasets."""
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
                    image_size, scale=(crop_scale_min, 1.0), ratio=(0.9, 1.1)
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


def build_breastdm_volume_transform(*, augment: bool = True) -> Optional[Callable]:
    """Build ADNI-style MONAI augmentation for a complete ``[1, D, H, W]`` volume."""
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
            "MONAI is required for BreastDM multi-slice augmentation; "
            "install MONAI or set augment=False"
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
        return transform({"image": volume})["image"]

    return apply


class _BreastDMBaseDataset(VisionDataset):
    """Base class that indexes one supplied BreastDM split under ``img17Se``."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
    ) -> None:
        normalized_split = str(split).strip().lower()
        root_path, image_root = _image_root(root)
        super().__init__(
            str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root_path
        self.image_root = image_root
        self.split = normalized_split
        self.class_names = BREASTDM_CLASS_NAMES
        self._records = _discover_records(image_root, normalized_split)
        self._volume_cache = _VolumeCache(volume_cache_size)
        logger.info(
            "Indexed BreastDM split=%s volumes=%d patients=%d labels=%s",
            self.split,
            len(self._records),
            len({record.patient_id for record in self._records}),
            {
                class_name: sum(record.label == label for record in self._records)
                for class_name, label in LABEL_TO_INDEX.items()
            },
        )

    @staticmethod
    def _metadata(record: _VolumeRecord) -> Dict[str, Any]:
        return {
            "split": record.split,
            "class_name": record.class_name,
            "patient_id": record.patient_id,
            "scan_id": record.scan_id,
            "volume_path": str(record.volume_path),
            "n_slices": record.n_slices,
        }


class BreastDMSingleSliceDataset(_BreastDMBaseDataset):
    """All individual 17-series BreastDM images from one supplied split.

    Each item returns ``(image, label)`` where ``Benign=0`` and ``Malignant=1``.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        augment: bool = True,
        image_size: int = 224,
        **kwargs: Any,
    ) -> None:
        if "transforms" not in kwargs and "transform" not in kwargs:
            kwargs["transform"] = build_breastdm_transform(
                image_size=image_size, augment=augment
            )
        super().__init__(root=root, split=split, **kwargs)
        self._entries = [
            (record, z) for record in self._records for z in range(record.n_slices)
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def get_target(self, index: int) -> int:
        return self._entries[index][0].label

    def get_patient_id(self, index: int) -> str:
        return self._entries[index][0].patient_id

    def get_scan_id(self, index: int) -> str:
        return self._entries[index][0].scan_id

    def get_slice_index(self, index: int) -> int:
        return self._entries[index][1]

    def get_volume_metadata(self, index: int) -> Dict[str, Any]:
        return self._metadata(self._entries[index][0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        image: Any = _slice_to_pil(self._volume_cache.get(record.volume_path), z)
        target: Any = record.label
        if self.transforms is not None:
            return self.transforms(image, target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class BreastDMPairedSliceDataset(BreastDMSingleSliceDataset):
    """Adjacent 17-series image pairs with shared stochastic 2-D augmentation."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._seed = seed
        super().__init__(root=root, **kwargs)

    def get_pair_slice_indices(self, index: int) -> Tuple[int, int]:
        record, z = self._entries[index]
        if record.n_slices < 2:
            return z, z
        if z == 0:
            return z, 1
        if z == record.n_slices - 1:
            return z, z - 1
        if self._seed is None:
            return z, random.choice((z - 1, z + 1))
        return z, random.Random(self._seed + int(index)).choice((z - 1, z + 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        _, partner_z = self.get_pair_slice_indices(index)
        volume = self._volume_cache.get(record.volume_path)
        images: Tuple[Any, Any] = (
            _slice_to_pil(volume, z),
            _slice_to_pil(volume, partner_z),
        )
        if self.transform is not None:
            return _apply_shared_pair_transform(self.transform, images)
        if self.transforms is not None:
            transformed = self.transforms(images, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return images


class BreastDMMultiSliceDataset(_BreastDMBaseDataset):
    """Full BreastDM volumes with optional shared 3-D augmentation.

    Each item returns ``([n_slices, 3, image_size, image_size], label)`` where
    labels are ``Benign=0`` and ``Malignant=1``.  The source depth is 17 and is
    resampled only when a different ``n_slices`` value is requested.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        n_slices: int = 17,
        image_size: int = 224,
        augment: bool = True,
        **kwargs: Any,
    ) -> None:
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.n_slices = int(n_slices)
        self.image_size = int(image_size)
        if "transforms" not in kwargs and "transform" not in kwargs and augment:
            kwargs["transform"] = build_breastdm_volume_transform(augment=True)
        super().__init__(root=root, split=split, **kwargs)

    def __len__(self) -> int:
        return len(self._records)

    def get_target(self, index: int) -> int:
        return self._records[index].label

    def get_patient_id(self, index: int) -> str:
        return self._records[index].patient_id

    def get_scan_id(self, index: int) -> str:
        return self._records[index].scan_id

    def get_volume_metadata(self, index: int) -> Dict[str, Any]:
        return self._metadata(self._records[index])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        record = self._records[index]
        image: Any = _resample_volume(
            _normalize_volume(self._volume_cache.get(record.volume_path)),
            self.n_slices,
            self.image_size,
        ).unsqueeze(0)
        target: Any = record.label
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        image = torch.as_tensor(image, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "BreastDM multi-slice transforms must return [1, depth, height, width], "
                f"got {tuple(image.shape)}"
            )
        return _volume_to_imagenet_tensors(image.squeeze(0)), target


# Short alias matching the naming used by the other classification datasets.
BreastDMSliceDataset = BreastDMSingleSliceDataset


__all__ = [
    "BREASTDM_CLASS_NAMES",
    "DEFAULT_ROOT",
    "SPLIT_CHOICES",
    "BreastDMMultiSliceDataset",
    "BreastDMPairedSliceDataset",
    "BreastDMSingleSliceDataset",
    "BreastDMSliceDataset",
    "build_breastdm_transform",
    "build_breastdm_volume_transform",
]
