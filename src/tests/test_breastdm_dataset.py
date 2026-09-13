from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from datasets.breastdm import (
    BreastDMMultiSliceDataset,
    BreastDMPairedSliceDataset,
    BreastDMSingleSliceDataset,
)


class TestBreastDMDataset(unittest.TestCase):
    @staticmethod
    def _write_volume(
        root: Path,
        split: str,
        class_name: str,
        patient_id: str,
        scan_id: str,
        offset: int,
    ) -> None:
        path = root / "img17Se" / split / class_name / patient_id
        path.mkdir(parents=True, exist_ok=True)
        image = np.broadcast_to(
            np.arange(17, dtype=np.uint8), (4, 5, 17)
        ).copy()
        np.save(path / f"{scan_id}.npy", image + offset)

    def _write_dataset(self, root: Path) -> None:
        self._write_volume(root, "train", "Benign", "B-001", "p-001", 0)
        self._write_volume(root, "train", "Malignant", "M-001", "p-002", 10)
        self._write_volume(root, "val", "Benign", "B-002", "p-003", 20)
        self._write_volume(root, "test", "Malignant", "M-002", "p-004", 30)
        # The dataset must never read the legacy nine-series directory.
        invalid = root / "img9Se" / "train" / "Benign" / "B-001"
        invalid.mkdir(parents=True)
        np.save(invalid / "ignored.npy", np.zeros((1, 1, 9), dtype=np.uint8))

    def test_single_slice_uses_supplied_split_and_binary_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            dataset = BreastDMSingleSliceDataset(
                root=root, split="train", augment=False, image_size=8
            )

            self.assertEqual(len(dataset), 34)
            self.assertEqual(dataset.class_names, ("Benign", "Malignant"))
            self.assertEqual(dataset.get_target(0), 0)
            self.assertEqual(dataset.get_target(17), 1)
            self.assertEqual(dataset.get_patient_id(0), "B-001")
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(target, 0)

    def test_pairs_are_adjacent_and_share_random_transform_state(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            dataset = BreastDMPairedSliceDataset(
                root=root,
                split="train",
                seed=4,
                transform=lambda _image: torch.rand(1),
            )

            for index in (0, 1, 8, 16, 17, 33):
                first, second = dataset.get_pair_slice_indices(index)
                self.assertLessEqual(abs(first - second), 1)
                self.assertNotEqual(first, second)
            first_view, second_view = dataset[8]
            torch.testing.assert_close(first_view, second_view)

    def test_multi_slice_returns_volume_level_label_and_requested_shape(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            dataset = BreastDMMultiSliceDataset(
                root=root,
                split="train",
                n_slices=4,
                image_size=8,
                augment=False,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual([dataset.get_target(index) for index in range(2)], [0, 1])
            volume, target = dataset[1]
            self.assertEqual(tuple(volume.shape), (4, 3, 8, 8))
            self.assertEqual(target, 1)
            self.assertEqual(dataset.get_volume_metadata(1)["patient_id"], "M-001")

    def test_multi_slice_three_d_encoder_returns_one_channel_volume(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            dataset = BreastDMMultiSliceDataset(
                root=root, split="train", n_slices=4, image_size=8,
                augment=False, three_d_encoder=True,
            )

            volume, target = dataset[1]

            self.assertEqual(tuple(volume.shape), (4, 1, 8, 8))
            self.assertEqual(target, 1)

    def test_rejects_unknown_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                BreastDMMultiSliceDataset(root=root, split="development", augment=False)
