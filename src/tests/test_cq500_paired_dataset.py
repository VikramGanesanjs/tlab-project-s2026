from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from datasets.cq500.dataset import (
    CQ500MultiSliceDataset,
    CQ500PairedSliceDataset,
    _VolumeRecord,
    _slice_to_pil,
    _window_volume_per_slice,
)
from utils.fold_cv import make_dataset_patient_folds


class TestCQ500PairedSliceDataset(unittest.TestCase):
    def test_slice_uses_fixed_brain_ct_window(self) -> None:
        volume = np.array([[[-100.0], [0.0], [40.0], [80.0], [200.0]]])

        image = _slice_to_pil(volume, 0)

        pixels = np.asarray(image)
        self.assertEqual(tuple(pixels.shape), (1, 5, 3))
        self.assertEqual(pixels[0, :, 0].tolist(), [0, 0, 127, 255, 255])
        self.assertTrue(np.array_equal(pixels[..., 0], pixels[..., 1]))
        self.assertTrue(np.array_equal(pixels[..., 1], pixels[..., 2]))

    def test_volume_windowing_is_independent_per_axial_slice(self) -> None:
        volume = np.array(
            [
                [[0.0, 40.0], [80.0, 120.0]],
                [[-20.0, 20.0], [60.0, 100.0]],
            ]
        )

        windowed = _window_volume_per_slice(volume)

        expected = np.array(
            [
                [[0.0, 0.5], [1.0, 1.0]],
                [[0.0, 0.25], [0.75, 1.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(windowed, expected)

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
                dataset[0]
                dataset[0]

        self.assertIsNone(dataset._volume_cache)
        self.assertEqual(load_volume.call_count, 2)


if __name__ == "__main__":
    unittest.main()
