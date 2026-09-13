from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from datasets.cq500.dataset import (
    CQ500MultiSliceDataset,
    CQ500PairedSliceDataset,
    _VolumeRecord,
    _slice_to_channels,
    _window_volume_per_slice,
)
from utils.fold_cv import make_dataset_patient_folds


class TestCQ500PairedSliceDataset(unittest.TestCase):
    def test_slice_uses_three_percentile_normalized_ct_windows(self) -> None:
        volume = np.array([[[-100.0], [0.0], [40.0], [80.0], [200.0]]])

        image = _slice_to_channels(volume, 0)

        self.assertEqual(tuple(image.shape), (3, 1, 5))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.all((0.0 <= image) & (image <= 1.0)))
        # Brain 40/80 clipping gives [0, 0, 40, 80, 80]; its 1st and 99th
        # percentiles are 0 and 80 respectively.
        torch.testing.assert_close(image[0, 0], torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
        self.assertFalse(torch.equal(image[0], image[1]))
        self.assertFalse(torch.equal(image[1], image[2]))

    def test_volume_windowing_is_independent_per_axial_slice(self) -> None:
        volume = np.array(
            [
                [[-1000.0, -500.0], [0.0, 80.0]],
                [[600.0, 1200.0], [2000.0, 2500.0]],
            ]
        )

        windowed = _window_volume_per_slice(volume)

        self.assertEqual(windowed.shape, (2, 3, 2, 2))
        self.assertTrue(np.all((0.0 <= windowed) & (windowed <= 1.0)))
        self.assertFalse(np.array_equal(windowed[:, 0], windowed[:, 1]))
        self.assertFalse(np.array_equal(windowed[:, 1], windowed[:, 2]))

    def _records(self, root: Path) -> list[_VolumeRecord]:
        return [
            _VolumeRecord(
                patient_id=f"CQ500CT{index}",
                scan_name="scan",
                volume_path=root / f"CQ500CT{index}" / "scan" / "volume.nii",
                n_slices=2,
                slice_spacing_mm=1.0,
                voxel_spacing_mm=(1.0, 1.0, 1.0),
            )
            for index in range(1, 7)
        ]

    def _write_labels(self, root: Path) -> Path:
        labels_path = root / "reads.csv"
        rows = [
            "name,R1:ICH,R1:IPH,R1:IVH,R1:SDH,R1:EDH,R1:SAH",
            "CQ500-CT-1,0,0,0,0,0,0",
            "CQ500-CT-2,1,1,0,0,0,0",
            "CQ500-CT-3,0,0,0,0,0,0",
            "CQ500-CT-4,1,0,1,0,0,0",
            "CQ500-CT-5,0,0,0,0,0,0",
            "CQ500-CT-6,1,0,0,1,0,0",
        ]
        labels_path.write_text("\n".join(rows), encoding="utf-8")
        return labels_path

    def test_ich_labels_enable_patient_stratified_folds(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root),
            ):
                dataset = CQ500PairedSliceDataset(
                    root=root,
                    csv_path=labels_path,
                    task="ich",
                    transform=lambda image: image,
                )

        self.assertEqual(len(dataset), 12)
        self.assertEqual(dataset.get_target(0), 0)
        self.assertEqual(dataset.get_target(2), 1)
        folds = make_dataset_patient_folds(dataset, n_folds=3, seed=4)
        self.assertEqual(folds.n_folds, 3)
        self.assertEqual(set(folds.patient_labels.values()), {0, 1})

    def test_subtype_task_filters_to_ich_positive_volumes_and_copies_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root),
            ):
                dataset = CQ500PairedSliceDataset(
                    root=root,
                    csv_path=labels_path,
                    task="subtype",
                    transform=lambda image: image,
                )

        self.assertEqual(len(dataset), 6)
        self.assertEqual(
            {dataset.get_patient_id(index) for index in range(len(dataset))},
            {"CQ500CT2", "CQ500CT4", "CQ500CT6"},
        )
        target = dataset.get_target(0)
        self.assertIsInstance(target, np.ndarray)
        target[0] = 0.0
        self.assertEqual(dataset.get_target(0)[0], 1.0)

    def test_patient_ids_restrict_paired_samples(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root),
            ):
                dataset = CQ500PairedSliceDataset(
                    root=root,
                    csv_path=labels_path,
                    patient_ids=["CQ500CT2", "CQ500CT5"],
                    transform=lambda image: image,
                )

        self.assertEqual(len(dataset), 4)
        self.assertEqual(
            {dataset.get_patient_id(index) for index in range(len(dataset))},
            {"CQ500CT2", "CQ500CT5"},
        )

    def test_paired_slices_are_three_channel_imagenet_normalized_tensors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            volume = np.array(
                [[[-1000.0, -500.0], [0.0, 80.0]], [[600.0, 1200.0], [2000.0, 2500.0]]],
                dtype=np.float32,
            )
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root)[:1],
            ):
                dataset = CQ500PairedSliceDataset(
                    root=root, csv_path=labels_path, augment=False, image_size=2, seed=3
                )
            with patch("datasets.cq500.dataset._load_canonical_volume", return_value=volume):
                image, partner = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 2, 2))
        self.assertEqual(tuple(partner.shape), (3, 2, 2))
        raw = _slice_to_channels(volume, 0)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        quantized = torch.round(raw * 255.0) / 255.0
        torch.testing.assert_close(image, (quantized - mean) / std)

    def test_paired_identity_transform_receives_quantized_rgb_pil_images(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            volume = np.linspace(-1000.0, 2500.0, num=8, dtype=np.float32).reshape(2, 2, 2)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root)[:1],
            ):
                dataset = CQ500PairedSliceDataset(
                    root=root, csv_path=labels_path, transform=lambda image: image, seed=3
                )
            with patch("datasets.cq500.dataset._load_canonical_volume", return_value=volume):
                image, partner = dataset[0]

        self.assertIsInstance(image, Image.Image)
        self.assertIsInstance(partner, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(np.asarray(image).dtype, np.uint8)

    def test_multi_slice_dataset_does_not_retain_full_volumes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            volume = np.linspace(-100.0, 120.0, num=8, dtype=np.float32).reshape(2, 2, 2)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root),
            ):
                dataset = CQ500MultiSliceDataset(
                    root=root,
                    csv_path=labels_path,
                    patient_ids=["CQ500CT1"],
                    n_slices=2,
                    image_size=2,
                    augment=False,
                )
            with patch(
                "datasets.cq500.dataset._load_canonical_volume", return_value=volume
            ) as load_volume:
                image, _ = dataset[0]
                dataset[0]

        self.assertIsNone(dataset._volume_cache)
        self.assertEqual(load_volume.call_count, 2)
        self.assertEqual(tuple(image.shape), (2, 3, 2, 2))
        # All three channels are ImageNet-normalized rather than grayscale copies.
        self.assertFalse(torch.equal(image[:, 0], image[:, 1]))
        self.assertFalse(torch.equal(image[:, 1], image[:, 2]))

    def test_multi_slice_three_d_encoder_returns_resampled_one_channel_volume(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = self._write_labels(root)
            volume = np.linspace(-100.0, 120.0, num=8, dtype=np.float32).reshape(2, 2, 2)
            with patch(
                "datasets.cq500.dataset._discover_volume_records",
                return_value=self._records(root),
            ):
                dataset = CQ500MultiSliceDataset(
                    root=root, csv_path=labels_path, patient_ids=["CQ500CT1"],
                    n_slices=2, image_size=2, augment=False, three_d_encoder=True,
                )
            with patch("datasets.cq500.dataset._load_canonical_volume", return_value=volume):
                image, _ = dataset[0]

        self.assertEqual(tuple(image.shape), (2, 1, 2, 2))
        self.assertTrue(torch.equal(image[:, 0], torch.from_numpy(volume).permute(2, 0, 1)))


if __name__ == "__main__":
    unittest.main()
