"""Adapters from tlab datasets to DINOv3's :class:`ExtendedVisionDataset`.

The underlying dataset classes remain the single source of modality-specific
loading logic.  This module only converts their raw slice outputs to PNG bytes
for DINOv3 and applies the common patient-level fold selection.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data.datasets.extended import ExtendedVisionDataset

from datasets.adni import ADNIClassificationDataset, DEFAULT_ROOT as ADNI_DEFAULT_ROOT
from datasets.amos import AMOSSingleSliceDataset, DEFAULT_ROOT as AMOS_DEFAULT_ROOT
from datasets.breastdm import (
    BreastDMSingleSliceDataset,
    DEFAULT_ROOT as BREASTDM_DEFAULT_ROOT,
)
from datasets.cq500 import CQ500SliceDataset, DEFAULT_ROOT as CQ500_DEFAULT_ROOT
from datasets.duke import DukeBreastMRIDataset
from datasets.duke.dataset import _DEFAULT_OUT_ROOT as DUKE_DEFAULT_ROOT
from datasets.organmnist3d import (
    DEFAULT_ROOT as ORGANMNIST3D_DEFAULT_ROOT,
    OrganMNIST3DClassificationDataset,
)
from datasets.lld_mmri import (
    DEFAULT_ROOT as LLD_MMRI_DEFAULT_ROOT,
    LLDMMRIClassificationDataset,
)
from utils.fold_cv import dataset_subset_for_patients, make_dataset_patient_folds

logger = logging.getLogger("dinov3")

DATASET_CHOICES = ("adni", "amos", "breastdm", "cq500", "duke", "lld_mmri", "organmnist3d")


def _identity_pair(image: Any, target: Any) -> tuple[Any, Any]:
    """Disable a dataset's default training transform while retaining raw slices."""
    return image, target


def _image_to_png(image: Any) -> bytes:
    """Serialize a PIL image, tensor, or NumPy slice as an RGB PNG."""
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim not in (2, 3):
            raise ValueError(f"Expected a 2-D or RGB image, got shape {array.shape}")
        array = np.asarray(array, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Cannot encode an image with no finite pixels")
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            high = low + 1.0
        pixels = np.nan_to_num(np.clip((array - low) / (high - low), 0.0, 1.0))
        pixels = np.rint(pixels * 255).astype(np.uint8)
        pil = Image.fromarray(pixels, mode="L" if pixels.ndim == 2 else "RGB").convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


class TlabExtendedVisionDataset(ExtendedVisionDataset):
    """Make an existing finite tlab slice dataset consumable by DINOv3."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        root: str | Path,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=str(Path(root).expanduser().resolve()),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_image_data(self, index: int) -> bytes:
        item = self.dataset[index]
        image = item[0] if isinstance(item, tuple) else item
        return _image_to_png(image)

    def get_target(self, index: int) -> Any:
        getter = getattr(self.dataset, "get_target", None)
        if callable(getter):
            return getter(index)
        item = self.dataset[index]
        return item[1] if isinstance(item, tuple) and len(item) > 1 else 0

    def get_patient_id(self, index: int) -> str:
        getter = getattr(self.dataset, "get_patient_id", None)
        if callable(getter):
            return str(getter(index))
        getter = getattr(self.dataset, "get_case_id", None)
        if callable(getter):
            return str(getter(index))
        getter = getattr(self.dataset, "get_volume_index", None)
        if callable(getter):
            return f"volume-{getter(index)}"
        return str(index)

    def get_image_relpath(self, index: int) -> str:
        return f"{type(self.dataset).__name__}/{self.get_patient_id(index)}/slice-{index}"


def build_base_dataset(
    dataset_name: str,
    *,
    root: Optional[str | Path] = None,
    task: Optional[str] = None,
    split: str = "train",
    csv_path: Optional[str | Path] = None,
    scan: Optional[Union[str, Sequence[str]]] = None,
) -> Dataset:
    """Build an untransformed single-slice dataset using :mod:`datasets`."""
    name = str(dataset_name).lower()
    if name == "adni":
        return ADNIClassificationDataset(
            root=root or ADNI_DEFAULT_ROOT, csv_path=csv_path, task=task or "cn_mci_ad",
            transforms=_identity_pair, augment=False,
        )
    if name == "lld_mmri":
        return LLDMMRIClassificationDataset(
            root=root or LLD_MMRI_DEFAULT_ROOT,
            scan_type="pre" if scan is None else scan,
            transforms=_identity_pair,
            augment=False,
        )
    if name == "cq500":
        return CQ500SliceDataset(
            root=root or CQ500_DEFAULT_ROOT, csv_path=csv_path, task=task or "ich",
            transforms=_identity_pair, augment=False,
        )
    if name == "breastdm":
        return BreastDMSingleSliceDataset(
            root=root or BREASTDM_DEFAULT_ROOT,
            split=split,
            transforms=_identity_pair,
            augment=False,
        )
    if name == "amos":
        return AMOSSingleSliceDataset(root=root or AMOS_DEFAULT_ROOT, transform=None)
    if name == "organmnist3d":
        return OrganMNIST3DClassificationDataset(
            root=root or ORGANMNIST3D_DEFAULT_ROOT, split=split, transform=None, augment=False,
        )
    if name == "duke":
        return DukeBreastMRIDataset(
            root=root or DUKE_DEFAULT_ROOT, return_pair=False, transforms=_identity_pair, augment=False,
        )
    raise ValueError(f"Unknown continued-pretraining dataset={dataset_name!r}; choose from {DATASET_CHOICES}")


def _fold_stratum(dataset: TlabExtendedVisionDataset, index: int) -> Any:
    target = dataset.get_target(index)
    if isinstance(target, np.ndarray):
        return tuple(int(value) for value in target.tolist())
    if isinstance(target, torch.Tensor):
        return tuple(int(value) for value in target.flatten().tolist())
    return target


def build_dataset_from_cfg(cfg: Any, *, transform: Callable, target_transform: Callable) -> Dataset:
    """Build and fold-filter a DINOv3-compatible dataset from ``cfg.train``.

    ``cfg.train.dataset`` enables this path.  Callers keep the legacy DINOv3
    ``dataset_path`` flow when that field is absent, preserving old configs.
    """
    train_cfg = cfg.train
    dataset_name = str(train_cfg.dataset).lower()
    root = getattr(train_cfg, "data_root", None)
    task = getattr(train_cfg, "task", None)
    csv_path = getattr(train_cfg, "csv_path", None)
    scan = getattr(train_cfg, "scan", None)
    split = getattr(train_cfg, "split", "train")
    base = build_base_dataset(
        dataset_name,
        root=root,
        task=task,
        split=split,
        csv_path=csv_path,
        scan=scan,
    )
    wrapped = TlabExtendedVisionDataset(
        base, root=root or getattr(base, "root_path", getattr(base, "root", ".")),
        transform=transform, target_transform=target_transform,
    )
    n_folds = int(getattr(train_cfg, "n_folds", 0))
    if n_folds == 0:
        return wrapped
    folds = make_dataset_patient_folds(
        wrapped, n_folds=n_folds, seed=int(getattr(train_cfg, "data_seed", getattr(train_cfg, "seed", 0))),
        target_fn=_fold_stratum,
    )
    train_ids, val_ids, test_ids = folds.get_split(
        int(getattr(train_cfg, "fold", 0)),
        train_ratio=float(getattr(train_cfg, "train_ratio", 1.0)),
        train_seed=int(getattr(train_cfg, "data_seed", getattr(train_cfg, "seed", 0))),
    )
    subset = dataset_subset_for_patients(wrapped, train_ids)
    logger.info(
        "%s continued pretraining fold %d/%d uses %d training patients (%d samples); reserved val/test=%d/%d",
        dataset_name, int(getattr(train_cfg, "fold", 0)), n_folds,
        len(train_ids), len(subset), len(val_ids), len(test_ids),
    )
    return subset


__all__ = ["DATASET_CHOICES", "TlabExtendedVisionDataset", "build_base_dataset", "build_dataset_from_cfg"]
