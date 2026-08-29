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

from datasets.amos import AMOSPairedSliceDataset, AMOSSingleSliceDataset


@unittest.skipIf(blosc2 is None, "blosc2 is required to create .b2nd fixtures")
class TestAMOSDatasets(unittest.TestCase):
    def _write_case(self, root: Path, case_id: str = "0001") -> None:
        image = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
        blosc2.asarray(image).save(str(root / f"{case_id}.b2nd"))

    def test_single_slice_indexes_every_z_plane_and_returns_rgb_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_case(root)
            dataset = AMOSSingleSliceDataset(root=root)

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.get_case_id(2), "0001")
            self.assertEqual(dataset.get_slice_index(2), 2)
            image = dataset[2]
            self.assertEqual(tuple(image.shape), (3, 3, 2))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.equal(image[0], image[1]))
            self.assertTrue(torch.equal(image[1], image[2]))

    def test_paired_slices_return_images_only_and_respect_distance(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_case(root)
            dataset = AMOSPairedSliceDataset(root=root, max_distance=1, seed=4)

            images = dataset[0]
            self.assertEqual(len(images), 2)
            self.assertEqual(tuple(images[0].shape), (3, 3, 2))
            self.assertEqual(tuple(images[1].shape), (3, 3, 2))
            self.assertLessEqual(dataset._partner_index(2, 2, 4) - 2, 1)
            self.assertGreaterEqual(dataset._partner_index(2, 2, 4) - 2, -1)

            alias = AMOSPairedSliceDataset(root=root, min_distance=1, seed=4)
            self.assertEqual(alias.max_distance, 1)
