from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from datasets.adni.dataset import (
    ADNIPairedSliceDataset,
    ADNIMultiSliceDataset,
    _ScanRecord,
    _scale_volume_to_unit_interval,
    _volume_to_imagenet_tensors,
)


class TestADNIVolumeToImageNetTensors(unittest.TestCase):
    def test_paired_dataset_restricts_samples_to_patient_ids(self) -> None:
        records = [
            _ScanRecord(
                image_id=f"I{index}",
                patient_id=f"patient-{index}",
                volume_path=Path(f"/unused/volume-{index}.nii"),
                n_slices=2,
                label=index,
                phenotype={"Group": "CN", "Sex": "F", "Age": "70"},
            )
            for index in range(2)
        ]
        with patch(
            "datasets.adni.dataset._build_scan_index",
            return_value=(records, ["Group", "Sex", "Age"]),
        ):
            dataset = ADNIPairedSliceDataset(
                root=Path("/unused"),
                patient_ids=["patient-1"],
                transform=lambda image: image,
            )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            {dataset.get_patient_id(index) for index in range(len(dataset))},
            {"patient-1"},
        )

    def test_applies_imagenet_normalization_without_per_slice_rescaling(self) -> None:
        volume = torch.tensor(
            [
                [[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]],
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
            ]
        )
        actual = _volume_to_imagenet_tensors(volume)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        expected = (volume.unsqueeze(1).repeat(1, 3, 1, 1) - mean) / std

        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (3, 3, 2, 3))
        self.assertTrue(torch.equal(actual, expected))

    def test_volume_scaling_uses_one_robust_scale_for_all_slices(self) -> None:
        volume = np.array([[[0.0, 10.0], [20.0, 1000.0]]], dtype=np.float32)
        scaled = _scale_volume_to_unit_interval(volume)

        self.assertEqual(scaled.dtype, np.float32)
        self.assertEqual(float(scaled.min()), 0.0)
        self.assertEqual(float(scaled.max()), 1.0)
        self.assertLess(float(scaled[0, 0, 1]), 0.1)

    def test_multi_slice_item_loads_without_using_the_volume_cache(self) -> None:
        dataset = object.__new__(ADNIMultiSliceDataset)
        dataset._entries = [
            _ScanRecord(
                image_id="I1",
                patient_id="patient-1",
                volume_path=Path("/unused/volume.nii"),
                n_slices=2,
                label=0,
                phenotype={},
            )
        ]
        dataset.n_slices = 2
        dataset.image_size = 2
        dataset.transforms = None
        dataset.transform = None
        dataset.target_transform = None
        dataset._volume_cache = object()

        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        with patch(
            "datasets.adni.dataset._load_canonical_volume", return_value=volume
        ) as load_volume:
            dataset[0]
            dataset[0]

        self.assertEqual(load_volume.call_count, 2)
