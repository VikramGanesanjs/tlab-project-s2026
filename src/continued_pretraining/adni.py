"""ADNI axial-slice adapter for DINOv3's unchanged data pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from dinov3.data.datasets.extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")
DEFAULT_CSV_NAME = "ADNI1_Complete_3Yr_1.5T_7_21_2026.csv"
_IMAGE_ID_RE = re.compile(r"^I[0-9]+$")
_DIAGNOSIS_TO_LABEL = {"CN": 0, "MCI": 1, "AD": 2}
_SPLITS = ("train", "val", "test")
_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


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


def _split_counts(n_patients: int) -> Dict[str, int]:
    """Allocate patients to train/validation/test using largest remainders."""
    expected = [_SPLIT_RATIOS[name] * n_patients for name in _SPLITS]
    counts = [math.floor(value) for value in expected]
    for index in sorted(
        range(len(_SPLITS)),
        key=lambda item: (-(expected[item] - counts[item]), item),
    )[: n_patients - sum(counts)]:
        counts[index] += 1
    return dict(zip(_SPLITS, counts))


def _make_patient_splits(patient_diagnoses: Mapping[str, str], seed: int) -> Dict[str, str]:
    """Create deterministic patient-level splits stratified by diagnosis."""
    rng = random.Random(seed)
    split_assignments: Dict[str, str] = {}
    for diagnosis in sorted(_DIAGNOSIS_TO_LABEL):
        patients = sorted(patient for patient, label in patient_diagnoses.items() if label == diagnosis)
        rng.shuffle(patients)
        counts = _split_counts(len(patients))
        start = 0
        for split in _SPLITS:
            end = start + counts[split]
            for patient in patients[start:end]:
                split_assignments[patient] = split
            start = end
    if set(split_assignments) != set(patient_diagnoses):
        raise RuntimeError("Failed to assign every ADNI patient to a split")
    return split_assignments


def _validate_patient_splits(
    split_assignments: Mapping[str, str],
    patient_diagnoses: Mapping[str, str],
) -> None:
    if set(split_assignments) != set(patient_diagnoses):
        missing = sorted(set(patient_diagnoses) - set(split_assignments))
        extra = sorted(set(split_assignments) - set(patient_diagnoses))
        raise ValueError(
            "Saved ADNI split file does not match the current dataset "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    invalid = {patient: split for patient, split in split_assignments.items() if split not in _SPLITS}
    if invalid:
        raise ValueError(f"Saved ADNI split file contains invalid split names: {invalid}")


def _patient_split_payload(
    split_assignments: Mapping[str, str],
    patient_diagnoses: Mapping[str, str],
    seed: int,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "ratios": dict(_SPLIT_RATIOS),
        "splits": {
            split: sorted(patient for patient, assigned_split in split_assignments.items() if assigned_split == split)
            for split in _SPLITS
        },
        "patient_diagnoses": dict(sorted(patient_diagnoses.items())),
    }


def _load_patient_splits(path: Path, patient_diagnoses: Mapping[str, str]) -> Dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    split_lists = payload.get("splits", payload)
    split_assignments = {
        patient: split
        for split in _SPLITS
        for patient in split_lists.get(split, [])
    }
    _validate_patient_splits(split_assignments, patient_diagnoses)
    saved_diagnoses = payload.get("patient_diagnoses")
    if saved_diagnoses is not None and dict(saved_diagnoses) != dict(patient_diagnoses):
        raise ValueError(f"Saved ADNI split file has different patient diagnoses: {path}")
    return split_assignments


class ADNI(ExtendedVisionDataset):
    """ADNI scans represented as canonical axial-slice images."""

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[str] = None,
        split: str = "TRAIN",
        split_seed: int = 0,
        split_file: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        split = str(split).strip().lower()
        if split not in _SPLITS:
            raise ValueError(f"Unknown ADNI split {split!r}; choose from {_SPLITS}")
        root_path = Path(root).expanduser().resolve()
        csv_path = Path(extra).expanduser().resolve() if extra else root_path / DEFAULT_CSV_NAME
        super().__init__(
            root=str(root_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._volume_cache = _VolumeCache()
        self.split = split
        self.split_seed = int(split_seed)
        self.split_file = Path(split_file).expanduser().resolve() if split_file else None
        entries, patient_diagnoses = self._build_index(root_path, csv_path)
        if self.split_file is not None and self.split_file.is_file():
            split_assignments = _load_patient_splits(self.split_file, patient_diagnoses)
        else:
            split_assignments = _make_patient_splits(patient_diagnoses, self.split_seed)
        self._patient_diagnoses = patient_diagnoses
        self._split_assignments = split_assignments
        self._entries = [
            (path, z, label, patient_id)
            for path, z, label, patient_id in entries
            if split_assignments[patient_id] == self.split
        ]
        if not self._entries:
            raise RuntimeError(f"No ADNI {self.split} slices found under {root_path}")
        patient_counts = {
            split_name: sum(assigned_split == split_name for assigned_split in split_assignments.values())
            for split_name in _SPLITS
        }
        diagnosis_counts = {
            diagnosis: sum(
                assigned_split == self.split and patient_diagnoses[patient] == diagnosis
                for patient, assigned_split in split_assignments.items()
            )
            for diagnosis in sorted(_DIAGNOSIS_TO_LABEL)
        }
        logger.info(
            "Indexed ADNI root=%s split=%s slices=%d patients=%d split_patient_counts=%s "
            "split_diagnoses=%s",
            root_path,
            self.split,
            len(self._entries),
            sum(assigned_split == self.split for assigned_split in split_assignments.values()),
            patient_counts,
            diagnosis_counts,
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

    def save_split_dictionary(self, path: Optional[str] = None) -> Path:
        """Persist the patient-to-split dictionary and return its path."""
        split_path = Path(path).expanduser().resolve() if path else self.split_file
        if split_path is None:
            raise ValueError("A split file path is required to save ADNI patient splits")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = split_path.with_name(f".{split_path.name}.{os.getpid()}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                _patient_split_payload(self._split_assignments, self._patient_diagnoses, self.split_seed),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary_path, split_path)
        logger.info("Saved ADNI patient split dictionary to %s", split_path)
        return split_path

    def get_image_data(self, index: int) -> bytes:
        path, z, _, _ = self._entries[index]
        return _slice_to_png(self._volume_cache.get(path), z)

    def get_target(self, index: int) -> Any:
        return self._entries[index][2]

    def get_image_relpath(self, index: int) -> str:
        path, z, _, _ = self._entries[index]
        return f"{path.relative_to(self.root)}#slice={z}"

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["ADNI"]
