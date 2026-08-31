from __future__ import annotations

import unittest

import torch

from datasets.adni.dataset import (
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
