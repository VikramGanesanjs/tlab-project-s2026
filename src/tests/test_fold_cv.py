import unittest

from utils.fold_cv import (
    make_dataset_patient_folds,
    make_patient_stratified_folds,
    select_class_balanced_patients,
)


class _ToyClassificationDataset:
    def __init__(self):
        self.entries = [
            (f"class-{label}-patient-{patient}", label)
            for label, count in ((0, 11), (1, 8), (2, 5))
            for patient in range(count)
            for _ in range(2)
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    def get_patient_id(self, index):
        return self.entries[index][0]

    def get_target(self, index):
        return self.entries[index][1]


class TestPatientStratifiedFolds(unittest.TestCase):
    def test_keeps_patients_together_and_balances_every_class(self):
        dataset = _ToyClassificationDataset()
        folds = make_dataset_patient_folds(dataset, n_folds=4, seed=19)

        assigned = [patient_id for fold in folds.folds for patient_id in fold]
        self.assertEqual(len(assigned), 24)
        self.assertEqual(len(set(assigned)), 24)
        for label in (0, 1, 2):
            counts = [
                sum(folds.patient_labels[patient_id] == label for patient_id in fold)
                for fold in folds.folds
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_get_split_returns_disjoint_train_validation_and_test_cases(self):
        patient_labels = {f"patient-{index:02d}": index % 3 for index in range(12)}
        folds = make_patient_stratified_folds(patient_labels, n_folds=4, seed=7)

        train_cases, val_cases, test_cases = folds.get_split(3)
        self.assertEqual(val_cases, list(folds.folds[3]))
        self.assertEqual(test_cases, list(folds.folds[0]))
        self.assertFalse(set(train_cases) & set(val_cases))
        self.assertFalse(set(train_cases) & set(test_cases))
        self.assertFalse(set(val_cases) & set(test_cases))
        self.assertEqual(set(train_cases + val_cases + test_cases), set(patient_labels))

    def test_is_reproducible_and_supports_an_explicit_test_fold(self):
        patient_labels = {f"patient-{index:02d}": index % 2 for index in range(16)}
        first = make_patient_stratified_folds(patient_labels, n_folds=4, seed=5)
        second = make_patient_stratified_folds(patient_labels, n_folds=4, seed=5)
        self.assertEqual(first, second)
        _train, val_cases, test_cases = first.get_cases(1, test_fold_index=3)
        self.assertEqual(val_cases, list(first.folds[1]))
        self.assertEqual(test_cases, list(first.folds[3]))

    def test_rejects_inconsistent_patient_labels(self):
        class InconsistentDataset(_ToyClassificationDataset):
            def __init__(self):
                self.entries = [("patient-a", 0), ("patient-a", 1), ("patient-b", 0)]

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            make_dataset_patient_folds(InconsistentDataset(), n_folds=3)

    def test_requires_at_least_three_folds(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            make_patient_stratified_folds({"a": 0, "b": 1}, n_folds=2)

    def test_train_ratio_selects_an_exact_class_balanced_subset(self):
        labels = {
            **{f"negative-{index}": 0 for index in range(9)},
            **{f"positive-{index}": 1 for index in range(3)},
        }
        selected = select_class_balanced_patients(
            list(labels), labels, train_ratio=0.5, seed=11
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            [sum(labels[patient] == label for patient in selected) for label in (0, 1)],
            [3, 3],
        )

    def test_get_split_applies_train_ratio_without_touching_holdouts(self):
        patient_labels = {f"patient-{index:02d}": index % 2 for index in range(20)}
        folds = make_patient_stratified_folds(patient_labels, n_folds=4, seed=7)
        full_train, val_cases, test_cases = folds.get_split(0)
        train_cases, ratio_val_cases, ratio_test_cases = folds.get_split(
            0, train_ratio=0.5
        )
        self.assertEqual(len(train_cases), len(full_train) // 2)
        self.assertEqual(val_cases, ratio_val_cases)
        self.assertEqual(test_cases, ratio_test_cases)
        self.assertTrue(set(train_cases).issubset(full_train))


if __name__ == "__main__":
    unittest.main()
