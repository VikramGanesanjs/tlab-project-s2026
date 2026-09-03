"""Self-supervised slice datasets backed by BraTSMen nnU-Net 2-D stores.

Each ``.b2nd`` image store has ``[channel, z, y, x]`` layout. BraTSMen has
four MRI modalities. A returned slice keeps all four channels, with each
modality independently min--max scaled to float values in the ``[0, 255]``
range before ImageNet normalization. This lets a training model learn its own
4-to-3 channel projection (for example with a 1x1 convolution). Segmentation
stores are intentionally ignored: these datasets are for SSL.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset

from datasets.amos.dataset import (
    _ImageNetNormalize,
    _RandomIntensityJitter,
    _scale_channels_to_255,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    _REPOSITORY_ROOT
    / "data"
    / "nnUNet_preprocessed"
    / "Dataset002_BraTSMen"
    / "nnUNetPlans_2d"
)


def _import_blosc2() -> Any:
    """Import the reader only when a BraTSMen dataset is instantiated."""
    try:
        import blosc2
    except ImportError as exc:
        raise ImportError(
            "BraTSMen datasets require the 'blosc2' package to read nnU-Net "
            ".b2nd files. Install blosc2 (it is included in the nnU-Net environment)."
        ) from exc
    return blosc2


def _apply_shared_pair_transform(
    transform: Callable, images: Tuple[Any, Any]
) -> Tuple[Any, Any]:
    """Apply one stochastic 2-D image augmentation realization to both slices."""
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
class _CaseRecord:
    case_id: str
    image_path: Path
    n_slices: int


class _BloscCaseCache:
    """Small process-local LRU cache of read-only Blosc2 image stores."""

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize <= 0:
            raise ValueError(f"volume_cache_size must be positive, got {maxsize}")
        self.maxsize = int(maxsize)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, record: _CaseRecord) -> Any:
        key = record.case_id
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        image = _import_blosc2().open(
            record.image_path,
            mode="r",
            mmap_mode="r",
            dparams={"nthreads": 1},
        )
        self._cache[key] = image
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return image


def _discover_cases(
    root: Path, case_ids: Optional[Sequence[Union[str, int]]]
) -> List[_CaseRecord]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"BraTSMen nnU-Net plans directory does not exist: {root}"
        )

    requested = (
        None if case_ids is None else {str(case_id).zfill(4) for case_id in case_ids}
    )
    image_paths = sorted(
        path for path in root.glob("*.b2nd") if not path.stem.endswith("_seg")
    )
    if requested is not None:
        image_paths = [path for path in image_paths if path.stem in requested]
        missing = requested.difference(path.stem for path in image_paths)
        if missing:
            raise FileNotFoundError(
                f"Requested BraTSMen case(s) not found under {root}: {sorted(missing)}"
            )
    if not image_paths:
        raise RuntimeError(f"No BraTSMen image .b2nd files found under {root}")

    blosc2 = _import_blosc2()
    records: List[_CaseRecord] = []
    for image_path in image_paths:
        # Reading metadata does not decompress the complete volume.
        image = blosc2.open(
            image_path, mode="r", mmap_mode="r", dparams={"nthreads": 1}
        )
        shape = tuple(image.shape)
        if len(shape) != 4 or shape[0] != 4:
            raise ValueError(
                f"Expected four-channel [4, z, y, x] image store for {image_path}, "
                f"got {shape}"
            )
        records.append(_CaseRecord(image_path.stem, image_path, int(shape[1])))
    return records


def _read_scaled_slice(
    record: _CaseRecord, z: int, cache: _BloscCaseCache
) -> torch.Tensor:
    """Read one axial MRI slice as four independently scaled float channels."""
    image_store = cache.get(record)
    image_2d = np.ascontiguousarray(np.asarray(image_store[:, z], dtype=np.float32))
    try:
        scaled = _scale_channels_to_255(image_2d)
    except ValueError as exc:
        raise ValueError(f"BraTSMen case {record.case_id} slice {z}: {exc}") from exc
    return torch.from_numpy(np.ascontiguousarray(scaled))


def build_brats_men_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.0,
) -> Callable:
    """Build ADNI-style 2-D augmentation for four-channel BraTSMen slices.

    Brightness/contrast jitter is applied with shared factors across the four
    MRI modalities. This is the four-channel equivalent of ADNI's
    ``ColorJitter``; hue and saturation do not have a medical-MRI analogue.
    ``augment=False`` performs deterministic square resizing followed by
    ImageNet normalization.
    """
    # Keep the validation and geometric augmentation choices identical to
    # AMOS, but reuse the generic tensor-safe jitter for four channels.
    from torchvision import transforms as tv_transforms

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
                _RandomIntensityJitter(jitter),
                tv_transforms.RandomRotation(rotation_degrees),
                tv_transforms.RandomHorizontalFlip(horizontal_flip_prob),
                tv_transforms.RandomVerticalFlip(vertical_flip_prob),
            ]
        )
    else:
        operations.append(tv_transforms.Resize((image_size, image_size)))
    operations.append(_ImageNetNormalize())
    return tv_transforms.Compose(operations)


class BraTSMenSingleSliceDataset(VisionDataset):
    """All axial BraTSMen slices for single-image self-supervised learning.

    Each z-plane of every volume is an item, so a shuffled DataLoader samples
    slices randomly. Items are float32 ImageNet-normalized tensors with shape
    ``[4, image_size, image_size]``; no segmentation target is read or
    returned.

    ``transform`` is an image-only callable. ``transforms`` is supported for
    consistency with :class:`VisionDataset`, called as ``transforms(image,
    None)``; its image result is returned.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        case_ids: Optional[Sequence[Union[str, int]]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        volume_cache_size: int = 4,
        augment: bool = True,
        image_size: int = 224,
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.root_path = Path(root).expanduser().resolve()
        self.image_size = int(image_size)
        if transforms is None and transform is None:
            transform = build_brats_men_transform(
                image_size=self.image_size, augment=augment
            )
        super().__init__(str(self.root_path), transforms=transforms, transform=transform)
        self._records = _discover_cases(self.root_path, case_ids)
        self._cache = _BloscCaseCache(volume_cache_size)
        self._entries: List[Tuple[_CaseRecord, int]] = [
            (record, z) for record in self._records for z in range(record.n_slices)
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def get_case_id(self, index: int) -> str:
        return self._entries[index][0].case_id

    def get_slice_index(self, index: int) -> int:
        return self._entries[index][1]

    def __getitem__(self, index: int) -> Any:
        record, z = self._entries[index]
        image: Any = _read_scaled_slice(record, z, self._cache)
        if self.transforms is not None:
            transformed = self.transforms(image, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return image


class BraTSMenPairedSliceDataset(BraTSMenSingleSliceDataset):
    """Random nearby slice pairs from one BraTSMen volume for SSL.

    Items are ``(slice_1, slice_2)`` four-channel float32 ImageNet-normalized
    tensors. The partner is selected from the same volume within the inclusive
    ``max_distance`` radius. At a one-slice boundary or with
    ``max_distance=0``, the indexed plane is reused.

    ``min_distance`` is an alias for the requested maximum offset; supplying
    it alongside ``max_distance`` is an error.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_ROOT,
        *,
        max_distance: Optional[int] = None,
        min_distance: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if max_distance is not None and min_distance is not None:
            raise ValueError("Specify only one of max_distance and min_distance")
        distance = 3 if max_distance is None and min_distance is None else (
            min_distance if max_distance is None else max_distance
        )
        if distance is None or distance < 0:
            raise ValueError(f"slice distance must be non-negative, got {distance}")
        self.max_distance = int(distance)
        self.seed = seed
        super().__init__(root=root, **kwargs)

    def _partner_index(self, index: int, z: int, n_slices: int) -> int:
        low = max(0, z - self.max_distance)
        high = min(n_slices - 1, z + self.max_distance)
        candidates = [candidate for candidate in range(low, high + 1) if candidate != z]
        if not candidates:
            return z
        if self.seed is None:
            return random.choice(candidates)
        return random.Random(self.seed + int(index)).choice(candidates)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        record, z = self._entries[index]
        partner_z = self._partner_index(index, z, record.n_slices)
        images: Tuple[Any, Any] = (
            _read_scaled_slice(record, z, self._cache),
            _read_scaled_slice(record, partner_z, self._cache),
        )
        if self.transform is not None:
            return _apply_shared_pair_transform(self.transform, images)
        if self.transforms is not None:
            transformed = self.transforms(images, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return images


# Concise alias for callers that do not need to distinguish the modality name.
BraTSMenSliceDataset = BraTSMenSingleSliceDataset


__all__ = [
    "BraTSMenPairedSliceDataset",
    "BraTSMenSingleSliceDataset",
    "BraTSMenSliceDataset",
    "DEFAULT_ROOT",
    "build_brats_men_transform",
]
