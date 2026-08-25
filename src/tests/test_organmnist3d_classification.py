from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from classification.model import MultiSliceDinoModel
from classification.run import parse_args
from classification.train import build_dataset, task_config, train


class _ToyEncoder(nn.Module):
    embed_dim = 8

    def forward_features(self, images: torch.Tensor):
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return {"x_norm_clstoken": pooled.repeat(1, self.embed_dim)}


class TestOrganMNIST3DClassificationIntegration(unittest.TestCase):
    def _write_archive(self, root: Path) -> None:
        labels = np.arange(11, dtype=np.uint8).reshape(-1, 1)

        def images(offset: int) -> np.ndarray:
            return np.full((11, 4, 4, 4), offset, dtype=np.uint8)

        np.savez(
            root / "organmnist3d_64.npz",
            train_images=images(32),
            train_labels=labels,
            val_images=images(64),
            val_labels=labels,
            test_images=images(96),
            test_labels=labels,
        )

    def test_official_splits_feed_the_multi_slice_classifier(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_archive(root)
            args = argparse.Namespace(
                dataset="organmnist3d",
                data_root=root,
                n_slices=2,
                image_size=8,
                augment=False,
            )
            datasets = {
                split: build_dataset(args, augment=False, split=split)
                for split in ("train", "val", "test")
            }
            self.assertEqual(
                {split: len(dataset) for split, dataset in datasets.items()},
                {"train": 11, "val": 11, "test": 11},
            )

            images, target = datasets["val"][0]
            self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
            self.assertEqual(target, 0)

            num_classes, class_names, loss_name = task_config("organmnist3d")
            self.assertEqual((num_classes, len(class_names), loss_name), (11, 11, "CrossEntropyLoss"))
            model = MultiSliceDinoModel(
                _ToyEncoder(),
                n_slices=2,
                features="cls",
                aggregator="transformer",
                d_model=8,
                depth=1,
                n_heads=2,
                ffn_dim=16,
                hidden_dim=8,
                num_classes=num_classes,
            )
            self.assertEqual(tuple(model(images.unsqueeze(0)).shape), (1, 11))

    def test_training_uses_official_splits_without_patient_partitioning(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_archive(root)
            args = parse_args(
                [
                    "--dataset",
                    "organmnist3d",
                    "--data-root",
                    str(root),
                    "--n-slices",
                    "2",
                    "--image-size",
                    "8",
                    "--no-augment",
                    "--batch-size",
                    "11",
                    "--num-workers",
                    "0",
                    "--epochs",
                    "1",
                    "--min-epochs",
                    "0",
                    "--no-early-stopping",
                    "--d-model",
                    "8",
                    "--mst-depth",
                    "1",
                    "--mst-heads",
                    "2",
                    "--mst-ffn-dim",
                    "16",
                    "--hidden-dim",
                    "8",
                ]
            )
            with patch("classification.train.load_encoder", return_value=_ToyEncoder()):
                with patch(
                    "classification.train.patient_level_stratified_split",
                    side_effect=AssertionError("official split path must not partition patients"),
                ):
                    train(args, torch.device("cpu"), root / "checkpoints")
            self.assertTrue((root / "checkpoints" / "best_mst.pt").is_file())
            self.assertTrue((root / "checkpoints" / "last_mst.pt").is_file())
