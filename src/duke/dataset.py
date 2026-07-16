"""PyTorch / DINOv3-compatible Duke Breast MRI dataset."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from .convert import (
    DEFAULT_PHENOTYPE_COLUMNS,
    PHENOTYPE_SENTINEL,
    _DEFAULT_OUT_ROOT,
    resolve_scan,
)

logger = logging.getLogger(__name__)

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


class DukeBreastMRIDataset(VisionDataset):
    """DINOv3-shaped Duke Breast MRI dataset.

    Each index is a primary axial slice from the selected scan. By default
    ``__getitem__`` also returns a second slice within ``max_distance``.
    """

    def __init__(
        self,
        root: Union[str, Path] = _DEFAULT_OUT_ROOT,
        *,
        scan: Union[str, int] = "post_1",
        max_distance: int = 3,
        return_pair: bool = True,
        split_breasts: bool = False,
        phenotype_columns: Optional[Sequence[str]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        volume_cache_size: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        root = Path(root)
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
            for z in range(n_z):
                self._entries.append((pid, vol_rel, n_z, z))

        if not self._entries:
            raise RuntimeError(
                f"No slices found for scan={self.scan_type!r} under {root}"
            )

        logger.info(
            "DukeBreastMRIDataset scan=%s entries=%d return_pair=%s max_distance=%d",
            self.scan_type,
            len(self._entries),
            self.return_pair,
            self.max_distance,
        )

    def __len__(self) -> int:
        return len(self._entries)

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

        target: Any = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                if self.return_pair:
                    image = (self.transform(image[0]), self.transform(image[1]))
                else:
                    image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target


# ---------------------------------------------------------------------------
# DINOv3 pair adapter
# ---------------------------------------------------------------------------


class PairToDinoGlobalCrops:
    """Wrap a single-image DINOv3 geometric/color transform for pair inputs.

    Expects ``image`` to be ``(pil_a, pil_b)``. Applies ``transform`` to each
    independently and returns a dict with ``global_crops`` (and copies teacher
    crops). Use with ``return_pair=True``.

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
        out_a = self.transform(img_a)
        out_b = self.transform(img_b)

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


