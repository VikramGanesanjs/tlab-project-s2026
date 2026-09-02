from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


class _ToyModel:
    def build_data_augmentation_dino(self, cfg):
        del cfg
        return object()


class TestCQ500SSLLoader(unittest.TestCase):
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
        build_dataset.assert_called_once_with(
            root=root,
            csv_path=root / "reads.csv",
            task="ich",
            max_distance=2,
            transform=train.identity_transform,
            image_size=224,
            seed=7,
        )
        self.assertEqual(len(make_loader.call_args.kwargs["dataset"]), 2)


if __name__ == "__main__":
    unittest.main()
