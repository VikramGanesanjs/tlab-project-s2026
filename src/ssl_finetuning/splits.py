"""Patient-level train/validation/test splitting for slice datasets."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Subset

logger = logging.getLogger("dinov3")

_SPLITS = ("train", "val", "test")


def _split_counts(n_patients: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    """Allocate patients by largest remainder while preserving the total."""
    expected = [fraction * n_patients for fraction in fractions]
    counts = [math.floor(value) for value in expected]
    remainder = n_patients - sum(counts)
    for index in sorted(
        range(len(counts)),
        key=lambda item: (-(expected[item] - counts[item]), item),
    )[:remainder]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _validate_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[float, float, float]:
    fractions = tuple(float(value) for value in (train_fraction, val_fraction, test_fraction))
    if any(not math.isfinite(value) or value < 0.0 for value in fractions):
        raise ValueError(f"Split fractions must be finite and non-negative, got {fractions}")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    if fractions[0] <= 0.0:
        raise ValueError("train_fraction must be greater than zero")
    return fractions


def _patient_stratum_key(value: Any) -> str:
    if value is None:
        return "__unknown__"
    return str(value)


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    dataset_name: str,
    patient_strata: Mapping[str, Any],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Dict[str, str]:
    if str(payload.get("dataset", "")).lower() != dataset_name.lower():
        raise ValueError(
            f"Split file dataset does not match: expected {dataset_name!r}, "
            f"got {payload.get('dataset')!r}"
        )
    saved_fractions = tuple(float(value) for value in payload.get("fractions", ()))
    if saved_fractions != fractions:
        raise ValueError(
            "Split file fractions do not match the current configuration: "
            f"saved={saved_fractions}, current={fractions}"
        )
    if int(payload.get("seed", seed)) != seed:
        raise ValueError(
            f"Split file seed does not match the current configuration: "
            f"saved={payload.get('seed')}, current={seed}"
        )

    saved_strata = payload.get("patient_strata", {})
    if saved_strata and dict(saved_strata) != dict(patient_strata):
        raise ValueError("Split file patient strata do not match the current dataset")

    split_lists = payload.get("splits")
    if not isinstance(split_lists, Mapping):
        raise ValueError("Split file is missing a 'splits' mapping")
    assignments: Dict[str, str] = {}
    for split in _SPLITS:
        patients = split_lists.get(split, [])
        if not isinstance(patients, list):
            raise ValueError(f"Split file entry {split!r} must be a list")
        for patient_id in patients:
            patient_id = str(patient_id)
            if patient_id in assignments:
                raise ValueError(f"Patient {patient_id!r} occurs in multiple splits")
            assignments[patient_id] = split

    expected_patients = set(patient_strata)
    if set(assignments) != expected_patients:
        raise ValueError(
            "Split file patients do not match the current dataset: "
            f"missing={sorted(expected_patients - set(assignments))[:5]}, "
            f"extra={sorted(set(assignments) - expected_patients)[:5]}"
        )
    return assignments


def patient_level_stratified_split(
    dataset: Dataset,
    *,
    dataset_name: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    stratum_fn: Callable[[Dataset, int], Any],
    split_file: Optional[Path] = None,
) -> Tuple[Subset, Dict[str, Any]]:
    """Return a train-only subset and the complete patient split metadata."""
    fractions = _validate_fractions(train_fraction, val_fraction, test_fraction)
    seed = int(seed)

    patient_indices: Dict[str, list[int]] = defaultdict(list)
    patient_strata: Dict[str, Any] = {}
    for index in range(len(dataset)):
        if not hasattr(dataset, "get_patient_id"):
            raise AttributeError(
                f"{type(dataset).__name__} must expose get_patient_id() for patient splitting"
            )
        patient_id = str(dataset.get_patient_id(index))  # type: ignore[attr-defined]
        patient_indices[patient_id].append(index)
        stratum = stratum_fn(dataset, index)
        if patient_id in patient_strata and patient_strata[patient_id] != stratum:
            raise ValueError(f"Patient {patient_id!r} has inconsistent split strata")
        patient_strata[patient_id] = stratum

    if not patient_indices:
        raise ValueError("Cannot split an empty dataset")

    if split_file is not None and split_file.is_file():
        with split_file.open(encoding="utf-8") as handle:
            assignments = _validate_payload(
                json.load(handle),
                dataset_name=dataset_name,
                patient_strata=patient_strata,
                fractions=fractions,
                seed=seed,
            )
    else:
        rng = random.Random(seed)
        grouped_patients: Dict[str, list[str]] = defaultdict(list)
        for patient_id, stratum in patient_strata.items():
            grouped_patients[_patient_stratum_key(stratum)].append(patient_id)

        split_patients: Dict[str, list[str]] = {split: [] for split in _SPLITS}
        for stratum, patients in sorted(grouped_patients.items()):
            patients = list(patients)
            rng.shuffle(patients)
            n_train, n_val, n_test = _split_counts(len(patients), fractions)
            boundaries = (n_train, n_train + n_val)
            split_patients["train"].extend(patients[: boundaries[0]])
            split_patients["val"].extend(patients[boundaries[0] : boundaries[1]])
            split_patients["test"].extend(patients[boundaries[1] : n_train + n_val + n_test])
            logger.info(
                "Patient split stratum=%s: train=%d val=%d test=%d",
                stratum,
                n_train,
                n_val,
                n_test,
            )

        # Ensure the training split is never empty, even for a very small
        # dataset whose rounded per-stratum allocations are pathological.
        if not split_patients["train"]:
            source = "val" if split_patients["val"] else "test"
            if not split_patients[source]:
                raise ValueError("Patient split produced no training patients")
            split_patients["train"].append(split_patients[source].pop())

        assignments = {
            patient_id: split
            for split, patients in split_patients.items()
            for patient_id in patients
        }

    # Sort the patient lists so an identical split file produces an identical
    # Subset ordering regardless of whether it was freshly generated or loaded.
    train_patients = sorted(patient for patient, split in assignments.items() if split == "train")
    val_patients = sorted(patient for patient, split in assignments.items() if split == "val")
    test_patients = sorted(patient for patient, split in assignments.items() if split == "test")
    train_indices = [index for patient in train_patients for index in patient_indices[patient]]
    val_indices = [index for patient in val_patients for index in patient_indices[patient]]
    test_indices = [index for patient in test_patients for index in patient_indices[patient]]

    if not set(train_patients).isdisjoint(val_patients + test_patients):
        raise RuntimeError("Patient leakage detected in train split")
    if set(val_patients).intersection(test_patients):
        raise RuntimeError("Patient leakage detected between validation and test splits")

    metadata: Dict[str, Any] = {
        "version": 1,
        "dataset": dataset_name,
        "seed": seed,
        "fractions": fractions,
        "patient_strata": dict(sorted(patient_strata.items())),
        "splits": {
            "train": sorted(train_patients),
            "val": sorted(val_patients),
            "test": sorted(test_patients),
        },
        "sample_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
    }
    logger.info(
        "Patient-level split for %s: train=%d patients/%d samples, "
        "val=%d patients/%d samples, test=%d patients/%d samples",
        dataset_name,
        len(train_patients),
        len(train_indices),
        len(val_patients),
        len(val_indices),
        len(test_patients),
        len(test_indices),
    )
    return Subset(dataset, train_indices), metadata


def save_patient_split(path: Path, metadata: Mapping[str, Any]) -> None:
    """Write split metadata atomically for reproducible future runs."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)
    logger.info("Saved patient split metadata to %s", path)


__all__ = ["patient_level_stratified_split", "save_patient_split"]
