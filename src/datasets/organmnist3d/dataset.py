"""PyTorch datasets for the 64³ OrganMNIST3D archive.

The official OrganMNIST3D ``.npz`` file already contains its train,
validation, and test partitions.  These datasets intentionally consume those
partitions directly rather than creating patient-level splits.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPOSITORY_ROOT / "data" / "OrganMNIST3D"
DEFAULT_NPZ_NAME = "organmnist3d_64.npz"
SPLIT_CHOICES = ("train", "val", "test")
ORGANMNIST3D_CLASS_NAMES = (
    "liver",
    "kidney-right",
    "kidney-left",
    "femur-right",
    "femur-left",
    "bladder",
    "heart",
    "lung-right",
    "lung-left",
    "spleen",
    "pancreas",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_image_size(image_size: int) -> int:
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    return image_size


def _resolve_npz_path(
    root: Union[str, Path], npz_path: Optional[Union[str, Path]]
) -> Tuple[Path, Path]:
    """Return ``(dataset_root, archive_path)`` for a directory or archive input."""
    if npz_path is not None:
        archive_path = Path(npz_path).expanduser().resolve()
        return archive_path.parent, archive_path

    supplied_path = Path(root).expanduser().resolve()
    if supplied_path.suffix == ".npz":
        return supplied_path.parent, supplied_path
    return supplied_path, supplied_path / DEFAULT_NPZ_NAME


def _load_split(archive_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    if split not in SPLIT_CHOICES:
        raise ValueError(f"Unknown split={split!r}; choose from {SPLIT_CHOICES}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing OrganMNIST3D archive: {archive_path}")

    image_key = f"{split}_images"
    label_key = f"{split}_labels"
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = {image_key, label_key}.difference(archive.files)
        if missing:
            raise ValueError(
                f"{archive_path} is missing required arrays: {sorted(missing)}; "
                f"found {archive.files}"
            )
        images = np.asarray(archive[image_key])
        labels = np.asarray(archive[label_key])

    if images.ndim != 4:
        raise ValueError(
            f"Expected {image_key} to have [N, D, H, W] layout, got {images.shape}"
        )
    if labels.ndim not in (1, 2) or labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"Expected one label per volume; got {image_key}={images.shape}, "
            f"{label_key}={labels.shape}"
        )
    return images, labels.reshape(-1).astype(np.int64, copy=False)


def _slice_to_pil(volume: np.ndarray, z: int) -> Image.Image:
    """Return a native-resolution RGB view of an OrganMNIST3D axial slice."""
    image_slice = np.asarray(volume[int(z)], dtype=np.uint8)
    return Image.fromarray(image_slice, mode="L").convert("RGB")


def _apply_shared_pair_transform(
    transform: Callable, images: Tuple[Any, Any]
) -> Tuple[Any, Any]:
    """Apply one stochastic 2-D augmentation realization to both slices."""
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


def build_organmnist3d_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
) -> tv_transforms.Compose:
    """Build the default 2-D preprocessing used by slice-based datasets."""
    image_size = _validate_image_size(image_size)
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

    operations = []
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


def build_organmnist3d_volume_transform(*, augment: bool = True) -> Optional[Callable]:
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
            "MONAI is required for OrganMNIST3D volume augmentation; "
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
        return transform({"image": volume})["image"]

    return apply


def _resize_volume_spatially(volume: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize only H/W, preserving the volume depth and ``[1, D, H, W]`` layout."""
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(
            "Expected a channel-first volume shaped [1, depth, height, width], "
            f"got {tuple(volume.shape)}"
        )
    return F.interpolate(
        volume.squeeze(0).unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).unsqueeze(0)


def _volume_to_imagenet_slices(volume: torch.Tensor) -> torch.Tensor:
    """Convert a normalized grayscale volume into ``[D, 3, H, W]`` tensors."""
    volume = torch.as_tensor(volume, dtype=torch.float32)
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(
            "OrganMNIST3D volume transforms must return [1, depth, height, width], "
            f"got shape {tuple(volume.shape)}"
        )
    slices = torch.nan_to_num(volume.squeeze(0), nan=0.0, posinf=1.0, neginf=0.0)
    slices = slices.clamp_(0.0, 1.0).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = slices.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = slices.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (slices - mean) / std


class _OrganMNIST3DBaseDataset(VisionDataset):
    """Base class that loads exactly one official OrganMNIST3D split."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        npz_path: Optional[Union[str, Path]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        normalized_split = str(split).strip().lower()
        root_path, archive_path = _resolve_npz_path(root, npz_path)
        images, labels = _load_split(archive_path, normalized_split)
        super().__init__(
            str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root_path
        self.npz_path = archive_path
        self.split = normalized_split
        self.images = images
        self.labels = labels
        self.depth = int(images.shape[1])

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def get_target(self, index: int) -> int:
        return int(self.labels[index])


class OrganMNIST3DClassificationDataset(_OrganMNIST3DBaseDataset):
    """Slice-level classification from the supplied OrganMNIST3D split."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        z_min: float = 0.0,
        z_max: float = 1.0,
        augment: bool = True,
        image_size: int = 224,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= z_min <= z_max <= 1.0:
            raise ValueError(
                f"Require 0 <= z_min <= z_max <= 1; got z_min={z_min}, z_max={z_max}"
            )
        if "transforms" not in kwargs and "transform" not in kwargs:
            kwargs["transform"] = build_organmnist3d_transform(
                image_size=image_size, augment=augment
            )
        super().__init__(root=root, split=split, **kwargs)
        start = int(np.floor(self.depth * float(z_min)))
        end = int(np.ceil(self.depth * float(z_max)))
        self.z_min, self.z_max = float(z_min), float(z_max)
        self._entries = [
            (volume_index, z)
            for volume_index in range(self.images.shape[0])
            for z in range(start, end)
        ]
        if not self._entries:
            raise RuntimeError(f"No slices selected by z-range [{z_min}, {z_max})")

    def __len__(self) -> int:
        return len(self._entries)

    def get_target(self, index: int) -> int:
        volume_index, _ = self._entries[index]
        return int(self.labels[volume_index])

    def get_volume_index(self, index: int) -> int:
        return int(self._entries[index][0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        volume_index, z = self._entries[index]
        image: Any = _slice_to_pil(self.images[volume_index], z)
        target: Any = int(self.labels[volume_index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return image, target


class OrganMNIST3DPairedSliceDataset(OrganMNIST3DClassificationDataset):
    """Pairs of nearby slices from the same OrganMNIST3D volume for SSL."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        max_distance: int = 3,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        self.max_distance = int(max_distance)
        self._seed = seed
        super().__init__(root=root, **kwargs)

    def _rng_for_index(self, index: int) -> np.random.RandomState:
        return np.random.RandomState() if self._seed is None else np.random.RandomState(self._seed + index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        volume_index, z = self._entries[index]
        start = int(np.floor(self.depth * self.z_min))
        end = int(np.ceil(self.depth * self.z_max))
        candidates = [partner for partner in range(max(start, z - self.max_distance), min(end - 1, z + self.max_distance) + 1) if partner != z]
        partner_z = int(self._rng_for_index(index).choice(candidates)) if candidates else z
        images: Tuple[Any, Any] = (
            _slice_to_pil(self.images[volume_index], z),
            _slice_to_pil(self.images[volume_index], partner_z),
        )
        if self.transform is not None:
            return _apply_shared_pair_transform(self.transform, images)
        if self.transforms is not None:
            transformed = self.transforms(images, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return images


class OrganMNIST3DMultiSliceDataset(_OrganMNIST3DBaseDataset):
    """Volume-level classification with optional 3-D encoder output.

    The entire native 64×64×64 volume is augmented first.  Its depth is then
    resampled to ``n_slices`` (64 by default) and each axial image is resized
    to ``image_size``; defaults therefore return ``[64, 3, 224, 224]``.
    Set ``three_d_encoder=True`` to return ``[depth, 1, height, width]``.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        split: str = "train",
        n_slices: int = 64,
        augment: bool = True,
        image_size: int = 224,
        three_d_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        self.n_slices = int(n_slices)
        self.image_size = _validate_image_size(image_size)
        self.three_d_encoder = bool(three_d_encoder)
        if "transforms" not in kwargs and "transform" not in kwargs and augment:
            kwargs["transform"] = build_organmnist3d_volume_transform(augment=True)
        super().__init__(root=root, split=split, **kwargs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        volume = np.ascontiguousarray(self.images[index], dtype=np.float32)
        volume_t: Any = torch.from_numpy(volume).unsqueeze(0).div_(255.0)
        target: Any = int(self.labels[index])
        if self.transforms is not None:
            volume_t, target = self.transforms(volume_t, target)
        else:
            if self.transform is not None:
                volume_t = self.transform(volume_t)
            if self.target_transform is not None:
                target = self.target_transform(target)

        volume_t = torch.as_tensor(volume_t, dtype=torch.float32)
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)
        if volume_t.ndim != 4 or volume_t.shape[0] != 1:
            raise ValueError(
                "OrganMNIST3D volume transforms must return [1, depth, height, width], "
                f"got shape {tuple(volume_t.shape)}"
            )
        if volume_t.shape[1] != self.n_slices:
            volume_t = F.interpolate(
                volume_t.unsqueeze(0),
                size=(self.n_slices, volume_t.shape[2], volume_t.shape[3]),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        volume_t = _resize_volume_spatially(volume_t, self.image_size)
        if self.three_d_encoder:
            return volume_t.permute(1, 0, 2, 3), target
        return _volume_to_imagenet_slices(volume_t), target


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
