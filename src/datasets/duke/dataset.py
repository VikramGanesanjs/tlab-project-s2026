"""PyTorch / DINOv3-compatible Duke Breast MRI dataset."""

from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.datasets.vision import VisionDataset

from .convert import (
    DEFAULT_PHENOTYPE_COLUMNS,
    PHENOTYPE_SENTINEL,
    _DEFAULT_OUT_ROOT,
    resolve_scan,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _VolumeCache:
    """Process-local LRU of memmapped NIfTI arrays."""

    def __init__(self, maxsize: int = 8) -> None:
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        import nibabel as nib

        img = nib.load(str(path), mmap=True)
        data = np.asanyarray(img.dataobj)  # memmap-backed when possible
        self._cache[key] = data
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return data


def _sample_partner_index(
    i: int,
    n_z: int,
    max_distance: int,
    rng: np.random.RandomState,
) -> int:
    if n_z < 2 or max_distance < 1:
        return i
    lo = max(0, i - max_distance)
    hi = min(n_z - 1, i + max_distance)
    candidates = [j for j in range(lo, hi + 1) if j != i]
    if not candidates:
        return i
    return int(rng.choice(candidates))


def _slice_to_pil(
    volume: np.ndarray,
    z: int,
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Image.Image:
    sl = np.asarray(volume[..., z], dtype=np.float32)
    lo, hi = np.percentile(sl, percentiles)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return Image.fromarray(u8, mode="L").convert("RGB")


def _split_breast_scaffold(
    image: Image.Image,
    patient_id: str,
    phenotype_raw: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Placeholder for left/right breast cropping via segmentation masks."""
    raise NotImplementedError(
        "split_breasts=True requires breast segmentation masks, which are not "
        f"available yet (patient={patient_id}). Set split_breasts=False."
    )


def _z_index_range(n_z: int, z_min: float, z_max: float) -> Tuple[int, int]:
    """Map fractional z thresholds in ``[0, 1]`` to a half-open index range.

    ``z_min`` / ``z_max`` are fractions of volume depth (0 = first slice, 1 =
    past the last). Middle 50% of slices → ``z_min=0.25``, ``z_max=0.75``.
    """
    if n_z <= 0:
        return 0, 0
    if not (0.0 <= z_min <= z_max <= 1.0):
        raise ValueError(
            f"Require 0 <= z_min <= z_max <= 1; got z_min={z_min}, z_max={z_max}"
        )
    start = int(np.floor(n_z * z_min))
    end = int(np.ceil(n_z * z_max))
    start = max(0, min(start, n_z))
    end = max(start, min(end, n_z))
    return start, end


def _zscore_normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize a whole volume using its nonzero voxel intensities."""
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    nonzero = volume[(volume > 0) & np.isfinite(volume)]
    if nonzero.size < 10:
        nonzero = finite
    if nonzero.size == 0:
        raise ValueError("Duke volume contains no finite voxel values")

    low, high = np.percentile(nonzero, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    volume = np.nan_to_num(volume, nan=0.0, posinf=high, neginf=0.0)
    volume = np.clip(volume, low, high)
    nonzero = volume[(volume > 0) & np.isfinite(volume)]
    if nonzero.size == 0:
        nonzero = volume.reshape(-1)
    mean = float(nonzero.mean())
    std = max(float(nonzero.std()), 1e-6)
    return ((volume - mean) / std).astype(np.float32, copy=False)


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


def build_duke_volume_transform(*, augment: bool = True) -> Optional[Callable]:
    """Build a MONAI transform for an entire Duke volume."""
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
            "MONAI is required for Duke multi-slice augmentation; "
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


def build_duke_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
) -> tv_transforms.Compose:
    """Build the default Duke preprocessing and augmentation pipeline."""
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


class DukeBreastMRIDataset(VisionDataset):
    """DINOv3-shaped Duke Breast MRI dataset.

    Each index is a primary axial slice from the selected scan. By default
    ``__getitem__`` also returns a second slice within ``max_distance``.

    ``z_min`` / ``z_max`` restrict which axial indices are indexed, as fractions
    of each volume's depth in ``[0, 1]`` (inclusive lower, exclusive upper after
    rounding). Use ``z_min=0.25``, ``z_max=0.75`` for the middle 50% of slices.
    """

    def __init__(
        self,
        root: Union[str, Path] = _DEFAULT_OUT_ROOT,
        *,
        scan: Union[str, int] = "pre",
        max_distance: int = 3,
        n_patients: Optional[int] = None,
        return_pair: bool = True,
        split_breasts: bool = False,
        z_min: float = 0.0,
        z_max: float = 1.0,
        phenotype_columns: Optional[Sequence[str]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        seed: Optional[int] = None,
        augment: bool = True,
        image_size: int = 224,
    ) -> None:
        root = Path(root)
        if transforms is None and transform is None:
            transform = build_duke_transform(image_size, augment=augment)
        super().__init__(
            str(root),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root
        self.scan_type = resolve_scan(scan)
        self.max_distance = int(max_distance)
        self.return_pair = bool(return_pair)
        self.split_breasts = bool(split_breasts)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.phenotype_columns = list(phenotype_columns or DEFAULT_PHENOTYPE_COLUMNS)
        self._seed = seed
        self._volume_cache = _VolumeCache(maxsize=volume_cache_size)

        index_path = root / "index" / "series_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Missing {index_path}. Run convert_duke_dataset() first."
            )
        with open(index_path) as f:
            series_index = json.load(f)

        pheno_path = root / "labels" / "phenotypes.json"
        schema_path = root / "labels" / "phenotype_schema.json"
        if not pheno_path.is_file():
            raise FileNotFoundError(f"Missing {pheno_path}")
        with open(pheno_path) as f:
            self._phenotypes = json.load(f)
        with open(schema_path) as f:
            self._phenotype_schema = json.load(f)

        schema_cols = self._phenotype_schema.get("columns", DEFAULT_PHENOTYPE_COLUMNS)
        self._col_indices = [schema_cols.index(c) for c in self.phenotype_columns]

        # Flat entries: (patient_id, rel_volume, n_slices, z)
        self._entries: List[Tuple[str, str, int, int]] = []
        for series in series_index.get("series", []):
            if series["scan_type"] != self.scan_type:
                continue
            pid = series["patient_id"]
            n_z = int(series["n_slices"])
            vol_rel = series["volume_path"]
            z_start, z_end = _z_index_range(n_z, self.z_min, self.z_max)
            for z in range(z_start, z_end):
                self._entries.append((pid, vol_rel, n_z, z))

        if not self._entries:
            raise RuntimeError(
                f"No slices found for scan={self.scan_type!r} under {root} "
                f"with z_min={self.z_min}, z_max={self.z_max}"
            )

        patient_ids = list(dict.fromkeys(entry[0] for entry in self._entries))
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
            entry for entry in self._entries if entry[0] in selected_patients
        ]
        self.n_patients = n_patients

        logger.info(
            "DukeBreastMRIDataset scan=%s entries=%d return_pair=%s "
            "max_distance=%d patients=%d z=[%.3f, %.3f)",
            self.scan_type,
            len(self._entries),
            self.return_pair,
            self.max_distance,
            self.n_patients,
            self.z_min,
            self.z_max,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def get_patient_id(self, index: int) -> str:
        """Return the patient identifier for patient-level dataset splits."""
        return self._entries[index][0]

    def get_target(self, index: int) -> np.ndarray:
        pid = self._entries[index][0]
        entry = self._phenotypes.get(pid)
        if entry is None:
            return np.full(len(self.phenotype_columns), PHENOTYPE_SENTINEL, dtype=np.float32)
        full = np.asarray(entry["vector"], dtype=np.float32)
        return full[self._col_indices].astype(np.float32)

    def get_phenotype_raw(self, index: int) -> Dict[str, Any]:
        pid = self._entries[index][0]
        entry = self._phenotypes.get(pid, {})
        return dict(entry.get("raw", {"patient_id": pid}))

    def _rng_for_index(self, index: int) -> np.random.RandomState:
        if self._seed is None:
            # Non-deterministic across workers/epochs for SSL diversity
            return np.random.RandomState()
        return np.random.RandomState(self._seed + int(index))

    def _load_slice(self, vol_rel: str, z: int, patient_id: str) -> Image.Image:
        volume = self._volume_cache.get(self.root_path / vol_rel)
        image = _slice_to_pil(volume, z)
        if self.split_breasts:
            image = _split_breast_scaffold(image, patient_id, self._phenotypes.get(patient_id, {}).get("raw"))
        return image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        patient_id, vol_rel, n_z, z = self._entries[index]
        img_i = self._load_slice(vol_rel, z, patient_id)

        if self.return_pair:
            j = _sample_partner_index(z, n_z, self.max_distance, self._rng_for_index(index))
            img_j = self._load_slice(vol_rel, j, patient_id)
            image: Any = (img_i, img_j)
        else:
            image = img_i

        if self.return_pair:
            if self.transform is not None:
                image = _apply_shared_pair_transform(self.transform, image)
            elif self.transforms is not None:
                transformed = self.transforms(image, None)
                image = transformed[0] if isinstance(transformed, tuple) else transformed
            return image

        target: Any = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        elif self.transforms is not None:
            image, target = self.transforms(image, target)
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def _normalize_laterality(value: Any) -> Optional[str]:
    """Map clinical tumor-location field to ``\"L\"`` / ``\"R\"``, else ``None``."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if v == PHENOTYPE_SENTINEL:
            return None
        if v == 0.0:
            return "L"
        if v == 1.0:
            return "R"
        return None
    s = str(value).strip().upper()
    if s in {"", "NA", "NC", "NP", "NAN", "NONE", "UNKNOWN"}:
        return None
    if s in {"L", "LEFT"}:
        return "L"
    if s in {"R", "RIGHT"}:
        return "R"
    return None


def _is_bilateral(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    s = str(value).strip().upper()
    return s in {"1", "YES", "TRUE", "BILATERAL"}


def _breast_cancer_label(side: str, tumor_location: Any, bilateral: Any) -> Optional[int]:
    """Binary cancer label for a breast side, or ``None`` if laterality is unknown."""
    loc = _normalize_laterality(tumor_location)
    if loc is None:
        return None
    if _is_bilateral(bilateral):
        return 1
    if side == "left":
        return 1 if loc == "L" else 0
    if side == "right":
        return 1 if loc == "R" else 0
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


class DukeClassificationDataset(VisionDataset):
    """Per-breast axial slice classification from left/right Duke MRI crops.

    Each index is one axial slice from a left or right breast crop volume.
    The target is a binary cancer label derived from clinical laterality
    (``tumor_location``) and the bilateral flag:

    * unknown laterality → patient excluded
    * bilateral → patient excluded by default (opt in with ``include_bilateral``)
    * unilateral L/R → only the matching side is cancerous (1); the other is 0

    ``z_min`` / ``z_max`` restrict axial indices as fractions of each crop's
    depth in ``[0, 1]`` (same semantics as :class:`DukeBreastMRIDataset`).
    """

    def __init__(
        self,
        root: Union[str, Path] = _DEFAULT_OUT_ROOT,
        *,
        scan: Union[str, int] = "pre",
        patient_ids: Optional[Sequence[str]] = None,
        z_min: float = 0.0,
        z_max: float = 1.0,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        include_bilateral: bool = False,
        build_default_transform: bool = True,
    ) -> None:
        root = Path(root)
        if build_default_transform and transforms is None and transform is None:
            transform = build_duke_transform(image_size, augment=augment)
        super().__init__(
            str(root),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_path = root
        self.scan_type = resolve_scan(scan)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.include_bilateral = bool(include_bilateral)
        self.patient_ids = (
            {str(patient_id) for patient_id in patient_ids}
            if patient_ids is not None
            else None
        )
        self._volume_cache = _VolumeCache(maxsize=volume_cache_size)

        index_path = root / "index" / "series_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Missing {index_path}. Run convert_duke_dataset() first."
            )
        with open(index_path) as f:
            series_index = json.load(f)

        pheno_path = root / "labels" / "phenotypes.json"
        if not pheno_path.is_file():
            raise FileNotFoundError(f"Missing {pheno_path}")
        with open(pheno_path) as f:
            self._phenotypes = json.load(f)

        # Flat entries: (patient_id, side, vol_rel, n_slices, z, label)
        self._entries: List[Tuple[str, str, str, int, int, int]] = []
        n_skipped_laterality = 0
        n_skipped_bilateral = 0
        n_skipped_missing_crops = 0

        for series in series_index.get("series", []):
            if series["scan_type"] != self.scan_type:
                continue
            pid = series["patient_id"]
            if self.patient_ids is not None and str(pid) not in self.patient_ids:
                continue
            raw = self._phenotypes.get(pid, {}).get("raw", {})
            if not self.include_bilateral and _is_bilateral(raw.get("bilateral")):
                n_skipped_bilateral += 1
                continue
            left_label = _breast_cancer_label("left", raw.get("tumor_location"), raw.get("bilateral"))
            if left_label is None:
                n_skipped_laterality += 1
                continue
            right_label = _breast_cancer_label(
                "right", raw.get("tumor_location"), raw.get("bilateral")
            )
            assert right_label is not None

            side_paths = self._resolve_breast_paths(series)
            if side_paths is None:
                n_skipped_missing_crops += 1
                continue

            for side, vol_rel, n_z, label in (
                ("left", side_paths["left"], side_paths["left_n_slices"], left_label),
                ("right", side_paths["right"], side_paths["right_n_slices"], right_label),
            ):
                z_start, z_end = _z_index_range(n_z, self.z_min, self.z_max)
                for z in range(z_start, z_end):
                    self._entries.append((pid, side, vol_rel, n_z, z, int(label)))

        if not self._entries:
            raise RuntimeError(
                f"No classification slices for scan={self.scan_type!r} under {root} "
                f"with z_min={self.z_min}, z_max={self.z_max} "
                f"(skipped_laterality={n_skipped_laterality}, "
                f"skipped_bilateral={n_skipped_bilateral}, "
                f"skipped_missing_crops={n_skipped_missing_crops})"
            )

        logger.info(
            "DukeClassificationDataset scan=%s entries=%d z=[%.3f, %.3f) "
            "include_bilateral=%s skipped_laterality=%d "
            "skipped_bilateral=%d skipped_missing_crops=%d",
            self.scan_type,
            len(self._entries),
            self.z_min,
            self.z_max,
            self.include_bilateral,
            n_skipped_laterality,
            n_skipped_bilateral,
            n_skipped_missing_crops,
        )

    def _resolve_breast_paths(
        self, series: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return left/right relative paths and slice counts, or ``None`` if missing."""
        pid = series["patient_id"]
        scan = series["scan_type"]
        left_rel = series.get("left_path") or f"{pid}/{scan}/left.nii"
        right_rel = series.get("right_path") or f"{pid}/{scan}/right.nii"
        left_path = self.root_path / left_rel
        right_path = self.root_path / right_rel
        if not left_path.is_file() or not right_path.is_file():
            return None

        left_n: Optional[int] = None
        right_n: Optional[int] = None
        meta_path = self.root_path / series.get("meta_path", f"{pid}/{scan}/meta.json")
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            crops = (meta.get("breast_divider") or {}).get("crops") or {}
            if "left" in crops and "shape" in crops["left"]:
                left_n = int(crops["left"]["shape"][-1])
            if "right" in crops and "shape" in crops["right"]:
                right_n = int(crops["right"]["shape"][-1])

        if left_n is None or right_n is None:
            import nibabel as nib

            if left_n is None:
                left_n = int(nib.load(str(left_path)).shape[-1])
            if right_n is None:
                right_n = int(nib.load(str(right_path)).shape[-1])

        return {
            "left": left_rel,
            "right": right_rel,
            "left_n_slices": left_n,
            "right_n_slices": right_n,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def get_target(self, index: int) -> int:
        return int(self._entries[index][5])

    def get_side(self, index: int) -> str:
        return self._entries[index][1]

    def get_phenotype_raw(self, index: int) -> Dict[str, Any]:
        pid = self._entries[index][0]
        entry = self._phenotypes.get(pid, {})
        return dict(entry.get("raw", {"patient_id": pid}))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _patient_id, _side, vol_rel, _n_z, z, label = self._entries[index]
        volume = self._volume_cache.get(self.root_path / vol_rel)
        image: Any = _slice_to_pil(volume, z)
        target: Any = int(label)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target


class DukeMultiSliceDataset(DukeClassificationDataset):
    """Per-breast classification using fixed-depth 3-D volumes.

    Each item represents one left or right breast volume from one patient.
    The complete volume is resampled to ``n_slices`` depth slices and spatially
    resized before an optional MONAI 3-D transform is applied.
    """

    def __init__(
        self,
        root: Union[str, Path] = _DEFAULT_OUT_ROOT,
        *,
        n_slices: int = 8,
        scan: Union[str, int] = "pre",
        patient_ids: Optional[Sequence[str]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        augment: bool = True,
        image_size: int = 224,
        include_bilateral: bool = False,
    ) -> None:
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        self.n_slices = int(n_slices)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if transforms is None and transform is None and augment:
            transform = build_duke_volume_transform(augment=True)
        super().__init__(
            root=root,
            scan=scan,
            patient_ids=patient_ids,
            z_min=0.0,
            z_max=1.0,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            volume_cache_size=volume_cache_size,
            augment=augment,
            image_size=image_size,
            build_default_transform=False,
            include_bilateral=include_bilateral,
        )

        # Collapse the parent's slice-level index to one entry per breast:
        # (patient_id, side, volume_path, volume_depth, label).
        volume_entries: List[Tuple[str, str, str, int, int]] = []
        seen = set()
        for pid, side, vol_rel, n_z, _z, label in self._entries:
            key = (pid, side, vol_rel)
            if key not in seen:
                seen.add(key)
                volume_entries.append((pid, side, vol_rel, n_z, label))
        self._entries = volume_entries  # type: ignore[assignment]
        logger.info(
            "DukeMultiSliceDataset scan=%s breasts=%d n_slices=%d image_size=%d",
            self.scan_type,
            len(self._entries),
            self.n_slices,
            self.image_size,
        )

    def get_target(self, index: int) -> int:
        return int(self._entries[index][4])

    def get_side(self, index: int) -> str:
        return self._entries[index][1]

    def get_patient_id(self, index: int) -> str:
        return self._entries[index][0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _pid, _side, vol_rel, _n_z, label = self._entries[index]
        volume = self._volume_cache.get(self.root_path / vol_rel)
        volume_t = _resample_volume(
            _zscore_normalize_volume(volume), self.n_slices, self.image_size
        ).unsqueeze(0)
        target: Any = int(label)

        if self.transforms is not None:
            volume_t, target = self.transforms(volume_t, target)
        else:
            if self.transform is not None:
                volume_t = self.transform(volume_t)
            if self.target_transform is not None:
                target = self.target_transform(target)

        volume_t = torch.as_tensor(volume_t)
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)
        if volume_t.ndim != 4 or volume_t.shape[0] != 1:
            raise ValueError(
                "Duke multi-slice transforms must return [1, depth, height, width], "
                f"got shape {tuple(volume_t.shape)}"
            )
        images = volume_t.permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        return images, target


# ---------------------------------------------------------------------------
# DINOv3 pair adapter
# ---------------------------------------------------------------------------


class PairToDinoGlobalCrops:
    """Wrap a single-image DINOv3 geometric/color transform for pair inputs.

    Expects ``image`` to be ``(pil_a, pil_b)``. Applies the same stochastic
    augmentation realization to both slices and returns a dict with
    ``global_crops`` (and copies teacher crops). Use with ``return_pair=True``.

    For classic DINOv3 SSL on a single slice, construct the dataset with
    ``return_pair=False`` and use stock ``DataAugmentationDINO`` directly.
    """

    def __init__(self, transform: Callable[[Image.Image], Any]) -> None:
        self.transform = transform

    def __call__(self, image: Any, target: Any = None) -> Any:
        if not (isinstance(image, (tuple, list)) and len(image) == 2):
            raise TypeError(
                "PairToDinoGlobalCrops expects image=(pil_a, pil_b); "
                "use return_pair=False with DataAugmentationDINO for single-image mode"
            )
        img_a, img_b = image
        out_a, out_b = _apply_shared_pair_transform(self.transform, (img_a, img_b))

        # If the wrapped transform is DataAugmentationDINO, it already returns a dict.
        # Take its first global crop from each slice as the two teacher/student views.
        if isinstance(out_a, dict) and "global_crops" in out_a:
            crop_a = out_a["global_crops"][0]
            crop_b = out_b["global_crops"][0]
            output = {
                "global_crops": [crop_a, crop_b],
                "global_crops_teacher": [crop_a, crop_b],
                "local_crops": out_a.get("local_crops", []) + out_b.get("local_crops", []),
                "offsets": out_a.get("offsets"),
                "weak_flag": out_a.get("weak_flag", True),
            }
            if target is None:
                return output
            return output, target

        # Plain tensor transform: treat outputs as the two global crops
        output = {
            "global_crops": [out_a, out_b],
            "global_crops_teacher": [out_a, out_b],
            "local_crops": [],
            "weak_flag": True,
        }
        if target is None:
            return output
        return output, target
