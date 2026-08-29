"""Self-supervised slice datasets backed by AMOS nnU-Net 2-D stores.

Each ``.b2nd`` image store has ``[channel, z, y, x]`` layout. AMOS contains a
single CT channel, which is copied into RGB for the DINO-style image encoder.
Segmentation stores are intentionally ignored: these datasets are for SSL.
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    _REPOSITORY_ROOT
    / "data"
    / "nnUNet_preprocessed"
    / "Dataset001_AMOS22"
    / "nnUNetPlans_2d"
)


def _import_blosc2() -> Any:
    """Import the reader only when an AMOS dataset is instantiated."""
    try:
        import blosc2
    except ImportError as exc:
        raise ImportError(
            "AMOS datasets require the 'blosc2' package to read nnU-Net .b2nd "
            "files. Install blosc2 (it is included in the nnU-Net environment)."
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
        raise FileNotFoundError(f"AMOS nnU-Net plans directory does not exist: {root}")

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
                f"Requested AMOS case(s) not found under {root}: {sorted(missing)}"
            )
    if not image_paths:
        raise RuntimeError(f"No AMOS image .b2nd files found under {root}")

    blosc2 = _import_blosc2()
    records: List[_CaseRecord] = []
    for image_path in image_paths:
        # Reading metadata does not decompress the complete volume.
        image = blosc2.open(
            image_path, mode="r", mmap_mode="r", dparams={"nthreads": 1}
        )
        shape = tuple(image.shape)
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(
                f"Expected one-channel [1, z, y, x] image store for {image_path}, "
                f"got {shape}"
            )
        records.append(_CaseRecord(image_path.stem, image_path, int(shape[1])))
    return records


def _read_rgb_slice(record: _CaseRecord, z: int, cache: _BloscCaseCache) -> torch.Tensor:
    """Read one axial CT slice and copy its single channel into RGB."""
    image_store = cache.get(record)
    image_2d = np.ascontiguousarray(np.asarray(image_store[0, z], dtype=np.float32))
    return torch.from_numpy(image_2d).unsqueeze(0).repeat(3, 1, 1)


class AMOSSingleSliceDataset(VisionDataset):
    """All axial AMOS slices for single-image self-supervised learning.

    Each z-plane of every volume is an item, so a shuffled DataLoader samples
    slices randomly. Items are RGB float32 tensors with shape ``[3, y, x]``;
    no segmentation target is read or returned.

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
    ) -> None:
        self.root_path = Path(root).expanduser().resolve()
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
        image: Any = _read_rgb_slice(record, z, self._cache)
        if self.transforms is not None:
            transformed = self.transforms(image, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return image


class AMOSPairedSliceDataset(AMOSSingleSliceDataset):
    """Random nearby slice pairs from one AMOS volume for SSL.

    Items are ``(slice_1, slice_2)`` RGB tensors. The partner is selected from
    the same volume within the inclusive ``max_distance`` radius. At a
    one-slice boundary or with ``max_distance=0``, the indexed plane is reused.

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
            _read_rgb_slice(record, z, self._cache),
            _read_rgb_slice(record, partner_z, self._cache),
        )
        if self.transform is not None:
            return _apply_shared_pair_transform(self.transform, images)
        if self.transforms is not None:
            transformed = self.transforms(images, None)
            return transformed[0] if isinstance(transformed, tuple) else transformed
        return images


# Concise alias for callers that do not need to distinguish the modality name.
AMOSSliceDataset = AMOSSingleSliceDataset


__all__ = [
    "AMOSPairedSliceDataset",
    "AMOSSingleSliceDataset",
    "AMOSSliceDataset",
    "DEFAULT_ROOT",
]
