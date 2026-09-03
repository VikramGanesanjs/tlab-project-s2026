from __future__ import annotations

import unittest

import torch

from utils.pca_dino_backbones import (
    _PCASingleSliceDataset,
    parse_args,
    parse_evolution_args,
)


class _FourChannelSliceDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        modalities = torch.stack(
            [torch.full((2, 2), float(channel)) for channel in range(4)]
        )
        mean = torch.tensor([0.485, 0.456, 0.406, 0.485]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229]).view(4, 1, 1)
        return (modalities - mean) / std

    def get_case_id(self, index: int) -> str:
        return "0001"

    def get_slice_index(self, index: int) -> int:
        return 3


class TestPCADinoBackbonesDatasetSupport(unittest.TestCase):
    def test_brats_men_adapter_projects_four_modalities_to_dino_rgb(self) -> None:
        dataset = _PCASingleSliceDataset(_FourChannelSliceDataset(), name="brats_men")

        image, target = dataset[0]

        self.assertIsNone(target)
        self.assertEqual(tuple(image.shape), (3, 2, 2))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        restored = image * std + mean
        torch.testing.assert_close(restored[0], torch.zeros(2, 2))
        torch.testing.assert_close(restored[1], torch.ones(2, 2))
        torch.testing.assert_close(restored[2], torch.full((2, 2), 2.5))
        self.assertEqual(dataset.get_patient_id(0), "0001")
        self.assertEqual(dataset.get_image_id(0), "0001_slice-3")

    def test_regular_pca_cli_accepts_new_dataset_names_without_data_root(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "brats_men",
                "--dinov3-checkpoint",
                "dino.pth",
                "--braindino-checkpoint",
                "brain.pth",
                "--custom-checkpoint",
                "custom.pth",
            ]
        )

        self.assertEqual(args.dataset, "brats_men")
        self.assertIsNone(args.data_root)

    def test_evolution_cli_accepts_all_new_single_slice_datasets(self) -> None:
        for dataset_name in ("cq500", "breastdm", "amos", "brats_men"):
            args = parse_evolution_args(
                ["--checkpoint-parent", "checkpoints", "--dataset", dataset_name]
            )
            self.assertEqual(args.dataset, dataset_name)
            self.assertIsNone(args.data_root)
