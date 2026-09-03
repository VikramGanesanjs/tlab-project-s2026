from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

try:
    import blosc2
except ImportError:  # pragma: no cover - depends on the selected Python environment
    blosc2 = None

from datasets.brats_men import BraTSMenPairedSliceDataset, BraTSMenSingleSliceDataset


@unittest.skipIf(blosc2 is None, "blosc2 is required to create .b2nd fixtures")
class TestBraTSMenDatasets(unittest.TestCase):
    def _write_case(self, root: Path, case_id: str = "0001") -> None:
        image = np.empty((4, 4, 2, 2), dtype=np.float32)
        for channel in range(4):
            for z in range(4):
                image[channel, z] = np.array(
                    [[10 * channel + z, 10 * channel + z + 1],
                     [10 * channel + z + 2, 10 * channel + z + 3]],
                    dtype=np.float32,
                )
        blosc2.asarray(image).save(str(root / f"{case_id}.b2nd"))

    def test_single_slice_indexes_every_z_plane_and_returns_normalized_channels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_case(root)
            dataset = BraTSMenSingleSliceDataset(root=root, image_size=2, augment=False)

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.get_case_id(2), "0001")
            self.assertEqual(dataset.get_slice_index(2), 2)
            image = dataset[2]
            self.assertEqual(tuple(image.shape), (4, 2, 2))
            self.assertEqual(image.dtype, torch.float32)
            expected = torch.tensor([[0.0, 85.0], [170.0, 255.0]]) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406, 0.485]).view(4, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.229]).view(4, 1, 1)
            expected = (expected.unsqueeze(0).repeat(4, 1, 1) - mean) / std
            self.assertTrue(torch.allclose(image, expected))

    def test_paired_slices_return_four_channel_images_and_respect_distance(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_case(root)
            dataset = BraTSMenPairedSliceDataset(
                root=root, max_distance=1, seed=4, image_size=2, augment=False
            )

            images = dataset[0]
            self.assertEqual(len(images), 2)
            self.assertEqual(tuple(images[0].shape), (4, 2, 2))
            self.assertEqual(tuple(images[1].shape), (4, 2, 2))
            self.assertEqual(images[0].dtype, torch.float32)
            self.assertEqual(images[1].dtype, torch.float32)
            self.assertLessEqual(dataset._partner_index(2, 2, 4) - 2, 1)
            self.assertGreaterEqual(dataset._partner_index(2, 2, 4) - 2, -1)

            alias = BraTSMenPairedSliceDataset(root=root, min_distance=1, seed=4)
            self.assertEqual(alias.max_distance, 1)

    def test_paired_augmentation_uses_the_same_parameters_for_both_slices(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_case(root)
            dataset = BraTSMenPairedSliceDataset(
                root=root, max_distance=0, image_size=5, augment=True
            )

            first, second = dataset[0]
            self.assertTrue(torch.equal(first, second))
