"""ADNI axial-slice adapter for DINOv3's unchanged data pipeline."""

from __future__ import annotations

import csv
import io
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from dinov3.data.datasets.extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")
DEFAULT_CSV_NAME = "ADNI1_Complete_3Yr_1.5T_7_21_2026.csv"
_IMAGE_ID_RE = re.compile(r"^I[0-9]+$")
_DIAGNOSIS_TO_LABEL = {"CN": 0, "MCI": 1, "AD": 2}


class _VolumeCache:
    def __init__(self, maxsize: int = 8) -> None:
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        import nibabel as nib

        volume = np.asanyarray(nib.as_closest_canonical(nib.load(key, mmap="r")).dataobj)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume at {path}, got {volume.shape}")
        self._cache[key] = volume
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return volume


def _slice_to_png(volume: np.ndarray, z: int) -> bytes:
    from PIL import Image

    image_slice = np.asarray(volume[..., z], dtype=np.float32)
    finite = image_slice[np.isfinite(image_slice)]
    if finite.size == 0:
        raise ValueError(f"Axial slice {z} contains no finite values")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    pixels = np.nan_to_num(
        np.clip((image_slice - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    image = Image.fromarray((pixels * 255.0).astype(np.uint8), mode="L").convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ADNI(ExtendedVisionDataset):
    """ADNI scans represented as canonical axial-slice images.

    Patient splitting is intentionally handled by
    :mod:`utils.splits`, which provides the shared split-file schema
    and validation used by both fine-tuning and continued pretraining.
    """

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        csv_path = Path(extra).expanduser().resolve() if extra else root_path / DEFAULT_CSV_NAME
        super().__init__(
            root=str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._volume_cache = _VolumeCache()
        entries, patient_diagnoses = self._build_index(root_path, csv_path)
        self._entries = entries
        if not self._entries:
            raise RuntimeError(f"No ADNI slices found under {root_path}")
        logger.info(
            "Indexed ADNI root=%s slices=%d patients=%d diagnoses=%s",
            root_path,
            len(self._entries),
            len(patient_diagnoses),
            {diagnosis: sum(label == diagnosis for label in patient_diagnoses.values())
             for diagnosis in sorted(_DIAGNOSIS_TO_LABEL)},
        )

    @staticmethod
    def _build_index(root: Path, csv_path: Path) -> tuple[list[tuple[Path, int, int, str]], Dict[str, str]]:
        metadata: Dict[str, Dict[str, str]] = {}
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Image Data ID", "Subject", "Group"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"ADNI CSV is missing required columns: {sorted(missing)}")
            for row in reader:
                image_id = str(row["Image Data ID"]).strip().upper()
                diagnosis = str(row["Group"]).strip().upper()
                if _IMAGE_ID_RE.fullmatch(image_id) and diagnosis in _DIAGNOSIS_TO_LABEL:
                    metadata[image_id] = row

        volumes: Dict[str, Path] = {}
        for path in sorted((*root.rglob("*.nii"), *root.rglob("*.nii.gz"))):
            image_id = path.parent.name.upper()
            if not _IMAGE_ID_RE.fullmatch(image_id):
                continue
            if image_id in volumes:
                raise ValueError(f"Multiple NIfTI files found for image ID {image_id}")
            volumes[image_id] = path

        entries = []
        patient_diagnoses: Dict[str, str] = {}
        for image_id in sorted(volumes, key=lambda item: int(item[1:])):
            if image_id not in metadata:
                raise ValueError(f"Missing ADNI CSV row for image ID {image_id}")
            import nibabel as nib

            image = nib.as_closest_canonical(nib.load(str(volumes[image_id]), mmap="r"))
            if len(image.shape) != 3:
                raise ValueError(f"Expected a 3D NIfTI volume at {volumes[image_id]}")
            row = metadata[image_id]
            patient_id = str(row["Subject"]).strip()
            diagnosis = str(row["Group"]).strip().upper()
            previous_diagnosis = patient_diagnoses.setdefault(patient_id, diagnosis)
            if previous_diagnosis != diagnosis:
                raise ValueError(
                    f"Patient {patient_id} has multiple diagnoses in the ADNI CSV: "
                    f"{previous_diagnosis} and {diagnosis}"
                )
            label = _DIAGNOSIS_TO_LABEL[diagnosis]
            entries.extend((volumes[image_id], z, label, patient_id) for z in range(int(image.shape[-1])))
        return entries, patient_diagnoses

    def get_image_data(self, index: int) -> bytes:
        path, z, _, _ = self._entries[index]
        return _slice_to_png(self._volume_cache.get(path), z)

    def get_target(self, index: int) -> Any:
        return self._entries[index][2]

    def get_patient_id(self, index: int) -> str:
        """Return the patient ID required by the shared split utility."""
        return self._entries[index][3]

    def get_image_relpath(self, index: int) -> str:
        path, z, _, _ = self._entries[index]
        return f"{path.relative_to(self.root)}#slice={z}"

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["ADNI"]
