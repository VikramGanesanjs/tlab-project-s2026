from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from ssl_finetuning import train


class _ToyCQ500Pairs:
    def __init__(self) -> None:
        self.patient_ids = [f"CQ500CT{index}" for index in range(6)]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def get_patient_id(self, index: int) -> str:
        return self.patient_ids[index]

    def get_target(self, index: int) -> int:
        return index % 2


class _ToyADNIPairs(_ToyCQ500Pairs):
    def get_phenotype_raw(self, index: int) -> dict[str, str]:
        return {"Group": ("CN", "MCI", "AD")[index % 3]}


class _ToyModel:
    def build_data_augmentation_dino(self, cfg):
        del cfg
        return object()


class TestCQ500SSLLoader(unittest.TestCase):
    def test_cross_slice_gram_penalty_delay_enables_at_boundary(self) -> None:
        delay_iterations = train._epochs_to_iterations(
            0.5,
            7,
            field_name="cross_slice_gram_penalty_delay_epochs",
        )
        self.assertEqual(delay_iterations, 4)
        self.assertEqual(train._delayed_penalty_weight(0.1, 3, 0), 0.0)
        self.assertEqual(train._delayed_penalty_weight(0.1, 3, 2), 0.0)
        self.assertEqual(train._delayed_penalty_weight(0.1, 3, 3), 0.1)
        self.assertEqual(train._delayed_penalty_weight(0.1, 0, 0), 0.1)

    def test_builds_adni_dataset_from_training_patient_ids(self) -> None:
        root = Path("/tmp/adni")
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                dataset="adni",
                data_root=root,
                adni_task="cn_mci_ad",
                max_distance=2,
                seed=7,
                n_folds=3,
                fold=0,
                data_seed=11,
                train_ratio=1.0,
                batch_size_per_gpu=2,
                num_workers=0,
                cache_dataset=True,
            ),
            crops=SimpleNamespace(global_crops_size=224),
        )
        dataset = _ToyADNIPairs()

        with (
            patch.object(train, "ADNIPairedSliceDataset", return_value=dataset) as build_dataset,
            patch.object(train, "PairToDinoGlobalCrops", return_value=object()),
            patch.object(train, "make_data_loader", return_value=object()),
        ):
            train.build_data_loader_from_cfg(cfg=cfg, model=_ToyModel(), start_iter=0)

        self.assertEqual(build_dataset.call_count, 2)
        self.assertEqual(len(build_dataset.call_args_list[1].kwargs["patient_ids"]), 2)
        self.assertTrue(
            set(build_dataset.call_args_list[1].kwargs["patient_ids"])
            .issubset(set(dataset.patient_ids))
        )
        self.assertNotIn("n_patients", build_dataset.call_args_list[0].kwargs)

    def test_builds_labeled_paired_cq500_dataset(self) -> None:
        root = Path("/tmp/cq500")
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                dataset="cq500",
                data_root=root,
                cq500_task="ich",
                max_distance=2,
                seed=7,
                n_folds=3,
                fold=0,
                data_seed=11,
                train_ratio=1.0,
                batch_size_per_gpu=2,
                num_workers=0,
                cache_dataset=True,
            ),
            crops=SimpleNamespace(global_crops_size=224),
        )
        dataset = _ToyCQ500Pairs()
        loader = object()

        with (
            patch.object(train, "CQ500PairedSliceDataset", return_value=dataset) as build_dataset,
            patch.object(train, "PairToDinoGlobalCrops", return_value=object()),
            patch.object(train, "make_data_loader", return_value=loader) as make_loader,
        ):
            actual_loader, size = train.build_data_loader_from_cfg(
                cfg=cfg,
                model=_ToyModel(),
                start_iter=0,
            )

        self.assertIs(actual_loader, loader)
        self.assertEqual(size, 2)
        self.assertEqual(build_dataset.call_count, 2)
        build_dataset.assert_has_calls([
            call(
                root=root,
                csv_path=root / "reads.csv",
                task="ich",
                max_distance=2,
                transform=train.identity_transform,
                image_size=224,
                seed=7,
            ),
            call(
                root=root,
                csv_path=root / "reads.csv",
                task="ich",
                patient_ids=build_dataset.call_args_list[1].kwargs["patient_ids"],
                max_distance=2,
                transform=train.identity_transform,
                image_size=224,
                seed=7,
            ),
        ])
        self.assertEqual(len(build_dataset.call_args_list[1].kwargs["patient_ids"]), 2)
        self.assertEqual(len(make_loader.call_args.kwargs["dataset"]), 2)

    def test_builds_breastdm_from_its_defined_training_split_without_folds(self) -> None:
        root = Path("/tmp/breastdm")
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                dataset="breastdm",
                data_root=root,
                seed=7,
                batch_size_per_gpu=2,
                num_workers=0,
                cache_dataset=True,
            ),
            crops=SimpleNamespace(global_crops_size=224),
        )
        dataset = _ToyCQ500Pairs()
        loader = object()

        with (
            patch.object(train, "BreastDMPairedSliceDataset", return_value=dataset) as build_dataset,
            patch.object(train, "_select_train_patient_ids") as select_patients,
            patch.object(train, "PairToDinoGlobalCrops", return_value=object()),
            patch.object(train, "make_data_loader", return_value=loader),
        ):
            actual_loader, size = train.build_data_loader_from_cfg(
                cfg=cfg,
                model=_ToyModel(),
                start_iter=0,
            )

        self.assertIs(actual_loader, loader)
        self.assertEqual(size, len(dataset))
        build_dataset.assert_called_once_with(
            root=root,
            split="train",
            transform=train.identity_transform,
            image_size=224,
            seed=7,
        )
        select_patients.assert_not_called()

    def test_builds_organmnist3d_from_its_official_training_split_without_folds(self) -> None:
        root = Path("/tmp/organmnist3d")
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                dataset="organmnist3d",
                data_root=root,
                max_distance=2,
                seed=7,
                batch_size_per_gpu=2,
                num_workers=0,
                cache_dataset=True,
            ),
            crops=SimpleNamespace(global_crops_size=224),
        )
        dataset = _ToyCQ500Pairs()
        loader = object()

        with (
            patch.object(
                train, "OrganMNIST3DPairedSliceDataset", return_value=dataset
            ) as build_dataset,
            patch.object(train, "_select_train_patient_ids") as select_patients,
            patch.object(train, "PairToDinoGlobalCrops", return_value=object()),
            patch.object(train, "make_data_loader", return_value=loader),
        ):
            actual_loader, size = train.build_data_loader_from_cfg(
                cfg=cfg,
                model=_ToyModel(),
                start_iter=0,
            )

        self.assertIs(actual_loader, loader)
        self.assertEqual(size, len(dataset))
        build_dataset.assert_called_once_with(
            root=root,
            split="train",
            max_distance=2,
            transform=train.identity_transform,
            image_size=224,
            seed=7,
        )
        select_patients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
