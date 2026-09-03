from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from datasets.adni.dataset import (
    ADNIMultiSliceDataset,
    _ScanRecord,
    _slice_to_imagenet_tensor,
    _volume_to_imagenet_tensors,
)


class TestADNIVolumeToImageNetTensors(unittest.TestCase):
    def test_matches_the_previous_per_slice_pil_conversion(self) -> None:
        volume = torch.tensor(
            [
                [[-2.0, -0.5, 0.0], [0.25, 0.5, 1.0]],
                [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
                [[-3.0, float("nan"), 1.0], [float("-inf"), 2.0, float("inf")]],
            ]
        )

        expected = torch.stack(
            [_slice_to_imagenet_tensor(image_slice) for image_slice in volume], dim=0
        )
        actual = _volume_to_imagenet_tensors(volume)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (3, 3, 2, 3))
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_a_slice_without_finite_values(self) -> None:
        volume = torch.tensor([[[0.0]], [[float("nan")]]])

        with self.assertRaisesRegex(ValueError, "contains no finite"):
            _volume_to_imagenet_tensors(volume)

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
