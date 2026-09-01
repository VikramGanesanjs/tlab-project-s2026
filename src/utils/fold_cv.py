"""Patient-level stratified folds for classification cross-validation.

The functions in this module operate on a dataset that exposes
``get_patient_id(index)`` and either ``get_target(index)`` or a caller-supplied
target function. All samples from one patient are assigned to the same fold,
while every class is distributed across folds as evenly as the number of
patients in that class permits.

For run ``i``, :meth:`PatientStratifiedFolds.get_split` uses fold ``i`` for
validation and fold ``(i + 1) % n_folds`` for testing.  This gives one simple,
deterministic train/validation/test split per fold index without ever putting a
patient in more than one split.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from torch.utils.data import Dataset, Subset


def _sorted_labels(labels: Iterable[Hashable]) -> list[Hashable]:
    """Sort potentially mixed-type labels reproducibly."""
    return sorted(labels, key=lambda label: (type(label).__name__, repr(label)))


def _validate_fold_count(n_folds: int, n_patients: int) -> int:
    if isinstance(n_folds, bool) or int(n_folds) != n_folds:
        raise ValueError(f"n_folds must be an integer, got {n_folds!r}")
    n_folds = int(n_folds)
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3 to leave out validation and test folds")
    if n_folds > n_patients:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of patients ({n_patients})"
        )
    return n_folds


def _balance_patient_groups(
    folds: list[list[str]],
    patient_labels: Mapping[str, Hashable],
    patient_groups: Mapping[str, Hashable],
) -> None:
    """Improve group balance with swaps that preserve label and fold counts."""
    n_folds = len(folds)
    group_totals: dict[Hashable, int] = defaultdict(int)
    group_counts: dict[Hashable, list[int]] = defaultdict(lambda: [0] * n_folds)
    for fold_index, fold in enumerate(folds):
        for patient_id in fold:
            group = patient_groups[patient_id]
            group_totals[group] += 1
            group_counts[group][fold_index] += 1
    group_targets = {group: total / n_folds for group, total in group_totals.items()}

    def swap_delta(
        first_fold: int,
        second_fold: int,
        first_group: Hashable,
        second_group: Hashable,
    ) -> float:
        if first_group == second_group:
            return 0.0
        delta = 0.0
        for group, first_change, second_change in (
            (first_group, -1, 1),
            (second_group, 1, -1),
        ):
            counts = group_counts[group]
            target = group_targets[group]
            before = (counts[first_fold] - target) ** 2 + (
                counts[second_fold] - target
            ) ** 2
            after = (counts[first_fold] + first_change - target) ** 2 + (
                counts[second_fold] + second_change - target
            ) ** 2
            delta += after - before
        return delta

    labels = _sorted_labels(set(patient_labels.values()))
    for _ in range(max(len(patient_labels) * 10, 1)):
        best: Optional[tuple[float, int, int, str, str]] = None
        for label in labels:
            for first_fold in range(n_folds):
                first_patients = sorted(
                    patient_id
                    for patient_id in folds[first_fold]
                    if patient_labels[patient_id] == label
                )
                for second_fold in range(first_fold + 1, n_folds):
                    second_patients = sorted(
                        patient_id
                        for patient_id in folds[second_fold]
                        if patient_labels[patient_id] == label
                    )
                    for first_patient in first_patients:
                        for second_patient in second_patients:
                            delta = swap_delta(
                                first_fold,
                                second_fold,
                                patient_groups[first_patient],
                                patient_groups[second_patient],
                            )
                            candidate = (
                                delta,
                                first_fold,
                                second_fold,
                                first_patient,
                                second_patient,
                            )
                            if delta < 0 and (best is None or candidate < best):
                                best = candidate
        if best is None:
            return
        _, first_fold, second_fold, first_patient, second_patient = best
        first_group = patient_groups[first_patient]
        second_group = patient_groups[second_patient]
        folds[first_fold].remove(first_patient)
        folds[first_fold].append(second_patient)
        folds[second_fold].remove(second_patient)
        folds[second_fold].append(first_patient)
        group_counts[first_group][first_fold] -= 1
        group_counts[first_group][second_fold] += 1
        group_counts[second_group][first_fold] += 1
        group_counts[second_group][second_fold] -= 1


@dataclass(frozen=True)
class PatientStratifiedFolds:
    """A reusable assignment of patients to stratified cross-validation folds.

    ``folds`` contains patient IDs only.  Thus, a patient with multiple scans
    or samples remains entirely within a single fold.
    """

    folds: tuple[tuple[str, ...], ...]
    patient_labels: Mapping[str, Hashable]
    seed: int

    @property
    def n_folds(self) -> int:
        """Number of available folds."""
        return len(self.folds)

    def get_split(
        self,
        fold_index: int,
        *,
        test_fold_index: Optional[int] = None,
        train_ratio: float = 1.0,
        train_seed: Optional[int] = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return ``(train_cases, val_cases, test_cases)`` for one CV run.

        ``fold_index`` identifies the validation fold.  By default, the next
        fold (wrapping around at the end) is used for testing, so callers only
        need to pass one index.  ``test_fold_index`` is available when a
        different validation/test pairing is needed. ``train_ratio`` selects a
        class-balanced fraction of the remaining training patients.
        """
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise ValueError(f"fold_index must be an integer, got {fold_index!r}")
        if not 0 <= fold_index < self.n_folds:
            raise ValueError(
                f"fold_index must be in [0, {self.n_folds - 1}], got {fold_index}"
            )

        if test_fold_index is None:
            test_fold_index = (fold_index + 1) % self.n_folds
        if isinstance(test_fold_index, bool) or not isinstance(test_fold_index, int):
            raise ValueError(
                f"test_fold_index must be an integer, got {test_fold_index!r}"
            )
        if not 0 <= test_fold_index < self.n_folds:
            raise ValueError(
                "test_fold_index must be in "
                f"[0, {self.n_folds - 1}], got {test_fold_index}"
            )
        if test_fold_index == fold_index:
            raise ValueError("validation and test folds must be different")

        train_cases = [
            patient_id
            for index, fold in enumerate(self.folds)
            if index not in (fold_index, test_fold_index)
            for patient_id in fold
        ]
        train_cases = select_class_balanced_patients(
            train_cases,
            self.patient_labels,
            train_ratio=train_ratio,
            seed=self.seed if train_seed is None else train_seed,
        )
        return train_cases, list(self.folds[fold_index]), list(self.folds[test_fold_index])

    # A descriptive alias for callers that prefer "cases" over "split".
    get_cases = get_split


def make_patient_stratified_folds(
    patient_labels: Mapping[str, Hashable],
    *,
    n_folds: int,
    seed: int = 0,
    patient_groups: Optional[Mapping[str, Hashable]] = None,
) -> PatientStratifiedFolds:
    """Create patient-level folds with per-class counts differing by at most one.

    Parameters
    ----------
    patient_labels:
        Mapping from a unique patient ID to its classification target.
    n_folds:
        Number of folds.  It must be at least three because every run reserves
        one fold for validation and another for testing.
    seed:
        Seed controlling the assignment of patients within each class.
    patient_groups:
        Optional secondary groups (for example, collection sites) balanced via
        within-class swaps after the primary label-stratified assignment.
    """
    normalized_labels = {str(patient_id): label for patient_id, label in patient_labels.items()}
    if len(normalized_labels) != len(patient_labels):
        raise ValueError("Patient IDs must remain unique after conversion to strings")
    if not normalized_labels:
        raise ValueError("Cannot create folds for an empty patient set")
    n_folds = _validate_fold_count(n_folds, len(normalized_labels))
    normalized_groups: Optional[dict[str, Hashable]] = None
    if patient_groups is not None:
        normalized_groups = {
            str(patient_id): group for patient_id, group in patient_groups.items()
        }
        if set(normalized_groups) != set(normalized_labels):
            raise ValueError("patient_groups must contain exactly the patient_labels IDs")
        for patient_id, group in normalized_groups.items():
            try:
                hash(group)
            except TypeError as error:
                raise ValueError(
                    f"Group for patient {patient_id!r} must be hashable, got {group!r}"
                ) from error

    patients_by_label: dict[Hashable, list[str]] = defaultdict(list)
    for patient_id, label in normalized_labels.items():
        try:
            hash(label)
        except TypeError as error:
            raise ValueError(
                f"Class label for patient {patient_id!r} must be hashable, got {label!r}"
            ) from error
        patients_by_label[label].append(patient_id)

    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Choosing the least populated fold as each class starts keeps overall fold
    # sizes balanced as well as the independently balanced class counts.
    for label in _sorted_labels(patients_by_label):
        patients = sorted(patients_by_label[label])
        rng.shuffle(patients)
        start = min(range(n_folds), key=lambda index: (len(folds[index]), index))
        for offset, patient_id in enumerate(patients):
            folds[(start + offset) % n_folds].append(patient_id)

    if normalized_groups is not None:
        _balance_patient_groups(folds, normalized_labels, normalized_groups)

    return PatientStratifiedFolds(
        folds=tuple(tuple(sorted(fold)) for fold in folds),
        patient_labels=dict(sorted(normalized_labels.items())),
        seed=int(seed),
    )


def make_dataset_patient_folds(
    dataset: Any,
    *,
    n_folds: int,
    seed: int = 0,
    patient_ids: Optional[Sequence[str]] = None,
    target_fn: Optional[Callable[[Any, int], Hashable]] = None,
    group_fn: Optional[Callable[[Any, int], Hashable]] = None,
) -> PatientStratifiedFolds:
    """Create folds for a classification dataset or a selected patient subset.

    Each patient's target (and optional group) must be consistent across all
    its samples.  Pass ``patient_ids`` to make folds from one pre-existing
    classification split rather than from every patient in ``dataset``.
    """
    get_patient_id = getattr(dataset, "get_patient_id", None)
    get_target = getattr(dataset, "get_target", None)
    if not callable(get_patient_id) or (target_fn is None and not callable(get_target)):
        raise TypeError(
            f"{type(dataset).__name__} must provide get_patient_id() and get_target(), "
            "or a target_fn must be supplied"
        )

    selected_patients = (
        {str(patient_id) for patient_id in patient_ids}
        if patient_ids is not None
        else None
    )
    patient_labels: dict[str, Hashable] = {}
    patient_groups: dict[str, Hashable] = {}
    for index in range(len(dataset)):
        patient_id = str(get_patient_id(index))
        if selected_patients is not None and patient_id not in selected_patients:
            continue
        label = target_fn(dataset, index) if target_fn is not None else get_target(index)
        try:
            hash(label)
        except TypeError as error:
            raise ValueError(
                f"Class label for patient {patient_id!r} must be hashable, got {label!r}"
            ) from error
        previous_label = patient_labels.setdefault(patient_id, label)
        if previous_label != label:
            raise ValueError(
                f"Patient {patient_id!r} has inconsistent class labels: "
                f"{previous_label!r} and {label!r}"
            )
        if group_fn is not None:
            group = group_fn(dataset, index)
            try:
                hash(group)
            except TypeError as error:
                raise ValueError(
                    f"Group for patient {patient_id!r} must be hashable, got {group!r}"
                ) from error
            previous_group = patient_groups.setdefault(patient_id, group)
            if previous_group != group:
                raise ValueError(
                    f"Patient {patient_id!r} has inconsistent groups: "
                    f"{previous_group!r} and {group!r}"
                )

    if selected_patients is not None:
        missing_patients = selected_patients - set(patient_labels)
        if missing_patients:
            raise ValueError(
                "Some requested patient IDs are absent from the dataset: "
                f"{sorted(missing_patients)[:5]}"
            )
    return make_patient_stratified_folds(
        patient_labels,
        n_folds=n_folds,
        seed=seed,
        patient_groups=patient_groups if group_fn is not None else None,
    )


def _validate_train_ratio(train_ratio: float) -> float:
    try:
        train_ratio = float(train_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError(f"train_ratio must be a finite value in (0, 1], got {train_ratio!r}") from error
    if not math.isfinite(train_ratio) or not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be a finite value in (0, 1], got {train_ratio!r}")
    return train_ratio


def _balanced_counts(
    grouped_patients: Mapping[Hashable, Sequence[str]],
    n_selected: int,
) -> dict[Hashable, int]:
    """Allocate an exact total as evenly as available class sizes permit."""
    labels = _sorted_labels(grouped_patients)
    if n_selected == 0:
        return {label: 0 for label in labels}

    # Allocate one patient per class at a time. This produces counts that are
    # equal (or differ by one) until a minority class is exhausted; only then
    # can the remaining slots be assigned to larger classes.
    counts = {label: 0 for label in labels}
    remaining = n_selected
    while remaining:
        eligible = [
            label for label in labels if counts[label] < len(grouped_patients[label])
        ]
        if not eligible:  # guarded by the caller, retained for clarity
            raise RuntimeError("Not enough patients for the requested train subset")
        lowest_count = min(counts[label] for label in eligible)
        for label in eligible:
            if counts[label] != lowest_count:
                continue
            counts[label] += 1
            remaining -= 1
            if remaining == 0:
                break
    return counts


def select_class_balanced_patients(
    patient_ids: Sequence[str],
    patient_labels: Mapping[str, Hashable],
    *,
    train_ratio: float,
    seed: int = 0,
) -> list[str]:
    """Select a deterministic, class-balanced fraction of training patients.

    The selected count is ``round(train_ratio * len(patient_ids))`` (with
    half values rounded up). Patients are drawn evenly across classes whenever
    class availability permits; when a minority class is exhausted, remaining
    slots are drawn from the other classes. The returned IDs are sorted so
    dataset order remains stable across runs.
    """
    train_ratio = _validate_train_ratio(train_ratio)
    unique_patient_ids = {str(patient_id) for patient_id in patient_ids}
    if len(unique_patient_ids) != len(patient_ids):
        raise ValueError("patient_ids must not contain duplicates")
    missing_labels = unique_patient_ids - set(patient_labels)
    if missing_labels:
        raise ValueError(
            "patient_labels is missing requested patient IDs: "
            f"{sorted(missing_labels)[:5]}"
        )
    n_available = len(unique_patient_ids)
    if not n_available:
        raise ValueError("Cannot select training patients from an empty set")
    n_selected = min(n_available, math.floor(train_ratio * n_available + 0.5))
    if n_selected == 0:
        # A positive ratio should still produce a usable training loader.
        n_selected = 1

    grouped_patients: dict[Hashable, list[str]] = defaultdict(list)
    for patient_id in unique_patient_ids:
        grouped_patients[patient_labels[patient_id]].append(patient_id)
    counts = _balanced_counts(grouped_patients, n_selected)

    rng = random.Random(int(seed))
    selected: list[str] = []
    for label in _sorted_labels(grouped_patients):
        candidates = sorted(grouped_patients[label])
        rng.shuffle(candidates)
        selected.extend(candidates[: counts[label]])
    return sorted(selected)


def dataset_subset_for_patients(dataset: "Dataset", patient_ids: Sequence[str]) -> "Subset":
    """Return every sample belonging to ``patient_ids`` as a stable subset."""
    get_patient_id = getattr(dataset, "get_patient_id", None)
    if not callable(get_patient_id):
        raise TypeError(f"{type(dataset).__name__} must provide get_patient_id()")
    selected = {str(patient_id) for patient_id in patient_ids}
    indices = [
        index for index in range(len(dataset)) if str(get_patient_id(index)) in selected
    ]
    found = {str(get_patient_id(index)) for index in indices}
    missing = selected - found
    if missing:
        raise ValueError(
            "Some requested patient IDs are absent from the dataset: "
            f"{sorted(missing)[:5]}"
        )
    from torch.utils.data import Subset

    return Subset(dataset, indices)


__all__ = [
    "PatientStratifiedFolds",
    "dataset_subset_for_patients",
    "make_dataset_patient_folds",
    "make_patient_stratified_folds",
    "select_class_balanced_patients",
]
