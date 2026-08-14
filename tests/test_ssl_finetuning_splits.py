from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from torch.utils.data import Dataset

from ssl_finetuning.splits import patient_level_stratified_split, save_patient_split


class _ToyPatientDataset(Dataset):
    def __init__(self, patients_per_stratum=10, samples_per_patient=2):
        self.entries = [
            (f"{stratum}-{patient}", stratum)
            for stratum in ("ad", "mci", "cn")
            for patient in range(patients_per_stratum)
            for _ in range(samples_per_patient)
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    def get_patient_id(self, index):
        return self.entries[index][0]


def _stratum(dataset, index):
    return dataset.entries[index][1]


def _split_strata(metadata):
    return {
        split: Counter(metadata["patient_strata"][patient] for patient in patients)
        for split, patients in metadata["splits"].items()
    }


class TestExplicitTrainPatientCount(unittest.TestCase):
    def test_uses_full_cohort_and_stratifies(self):
        dataset = _ToyPatientDataset()
        train_dataset, metadata = patient_level_stratified_split(
            dataset,
            dataset_name="toy",
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            train_patient_count=12,
            seed=3,
            stratum_fn=_stratum,
        )

        self.assertEqual(
            {split: len(patients) for split, patients in metadata["splits"].items()},
            {"train": 12, "val": 9, "test": 9},
        )
        self.assertEqual(len(train_dataset), 24)
        self.assertEqual(
            _split_strata(metadata),
            {
                "train": Counter({"ad": 4, "mci": 4, "cn": 4}),
                "val": Counter({"ad": 3, "mci": 3, "cn": 3}),
                "test": Counter({"ad": 3, "mci": 3, "cn": 3}),
            },
        )
        assigned = [
            patient for patients in metadata["splits"].values() for patient in patients
        ]
        self.assertEqual(len(assigned), 30)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_is_exact_and_nearly_balanced_with_rounding(self):
        dataset = _ToyPatientDataset()
        _, metadata = patient_level_stratified_split(
            dataset,
            dataset_name="toy",
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            train_patient_count=13,
            seed=3,
            stratum_fn=_stratum,
        )

        strata = _split_strata(metadata)
        self.assertEqual(len(metadata["splits"]["train"]), 13)
        for split in ("train", "val", "test"):
            self.assertLessEqual(max(strata[split].values()) - min(strata[split].values()), 1)

    def test_saved_split_requires_the_same_explicit_train_count(self):
        dataset = _ToyPatientDataset()
        with TemporaryDirectory() as temporary_directory:
            split_path = Path(temporary_directory) / "toy_splits.json"
            _, metadata = patient_level_stratified_split(
                dataset,
                dataset_name="toy",
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                train_patient_count=12,
                seed=3,
                stratum_fn=_stratum,
            )
            save_patient_split(split_path, metadata)

            _, loaded_metadata = patient_level_stratified_split(
                dataset,
                dataset_name="toy",
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                train_patient_count=12,
                seed=3,
                stratum_fn=_stratum,
                split_file=split_path,
            )
            self.assertEqual(metadata, loaded_metadata)

            with self.assertRaisesRegex(ValueError, "train_patient_count"):
                patient_level_stratified_split(
                    dataset,
                    dataset_name="toy",
                    train_fraction=0.70,
                    val_fraction=0.15,
                    test_fraction=0.15,
                    train_patient_count=13,
                    seed=3,
                    stratum_fn=_stratum,
                    split_file=split_path,
                )

    def test_saved_split_can_ignore_explicit_train_count_configuration(self):
        dataset = _ToyPatientDataset()
        with TemporaryDirectory() as temporary_directory:
            split_path = Path(temporary_directory) / "toy_splits.json"
            _, metadata = patient_level_stratified_split(
                dataset,
                dataset_name="toy",
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                train_patient_count=12,
                seed=3,
                stratum_fn=_stratum,
            )
            save_patient_split(split_path, metadata)

            _, loaded_metadata = patient_level_stratified_split(
                dataset,
                dataset_name="toy",
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                train_patient_count=None,
                seed=3,
                stratum_fn=_stratum,
                split_file=split_path,
                use_saved_split_config=False,
            )

        self.assertEqual(loaded_metadata["splits"], metadata["splits"])

    def test_loads_legacy_continued_pretraining_adni_split(self):
        dataset = _ToyPatientDataset()
        _, metadata = patient_level_stratified_split(
            dataset,
            dataset_name="adni",
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            seed=3,
            stratum_fn=_stratum,
        )
        legacy_payload = {
            "seed": 3,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "patient_diagnoses": dict(metadata["patient_strata"]),
            "splits": metadata["splits"],
        }
        with TemporaryDirectory() as temporary_directory:
            split_path = Path(temporary_directory) / "adni_splits.json"
            split_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
            _, loaded_metadata = patient_level_stratified_split(
                dataset,
                dataset_name="adni",
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                seed=3,
                stratum_fn=_stratum,
                split_file=split_path,
                use_saved_split_config=False,
                validate_saved_patient_strata=False,
            )

        self.assertEqual(loaded_metadata["splits"], metadata["splits"])


if __name__ == "__main__":
    unittest.main()
