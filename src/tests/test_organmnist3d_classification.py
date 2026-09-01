from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from classification.model import MultiSliceDinoModel
from classification.run import parse_args
from classification.train import build_dataset, saved_split_patient_ids, task_config, train
from utils import load_dinov3


class _ToyEncoder(nn.Module):
    embed_dim = 8

    def forward_features(self, images: torch.Tensor):
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return {"x_norm_clstoken": pooled.repeat(1, self.embed_dim)}


class _ToyPatientDataset:
    def __init__(self, patient_ids: list[str]) -> None:
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def get_patient_id(self, index: int) -> str:
        return self.patient_ids[index]

    def get_target(self, index: int) -> int:
        return index % 2


class TestOrganMNIST3DClassificationIntegration(unittest.TestCase):
    def test_meddinov3_is_a_selectable_classification_encoder(self) -> None:
        args = parse_args(["--encoder", "meddinov3"])

        self.assertEqual(args.encoder, "meddinov3")
        self.assertEqual(
            load_dinov3.DEFAULT_ENCODER_WEIGHTS["meddinov3"],
            load_dinov3.REPO_ROOT / "opt" / "meddinov3" / "model.pth",
        )

    def test_meddinov3_dispatches_to_the_teacher_checkpoint_loader(self) -> None:
        sentinel = object()
        with patch.object(
            load_dinov3,
            "load_meddinov3_encoder",
            return_value=sentinel,
        ) as loader:
            encoder = load_dinov3.load_encoder("meddinov3", device=torch.device("cpu"))

        self.assertIs(encoder, sentinel)
        loader.assert_called_once_with(
            repo_dir=load_dinov3.DINOV3_REPO,
            weights=load_dinov3.DEFAULT_MEDDINOV3_WEIGHTS,
            device=torch.device("cpu"),
        )

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

    def test_native_depth_transformer_ignores_padded_slices(self) -> None:
        """A native-depth CQ500 batch can use the transformer aggregator."""
        model = MultiSliceDinoModel(
            _ToyEncoder(),
            n_slices=None,
            features="cls",
            aggregator="transformer",
            d_model=8,
            depth=1,
            n_heads=2,
            ffn_dim=16,
            dropout=0.0,
            hidden_dim=8,
        ).eval()
        short_volume = torch.randn(2, 3, 4, 4)
        long_volume = torch.randn(4, 3, 4, 4)
        padded_batch = torch.zeros(2, 4, 3, 4, 4)
        padded_batch[0, :2] = short_volume
        padded_batch[1] = long_volume
        slice_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

        with torch.no_grad():
            standalone = model(short_volume.unsqueeze(0))
            batched = model(padded_batch, slice_mask=slice_mask)

        self.assertIsNone(model.position_embedding)
        torch.testing.assert_close(batched[0], standalone[0])

    def test_cq500_null_depth_accepts_transformer_aggregator(self) -> None:
        args = parse_args(["--dataset", "cq500", "--n-slices", "null"])

        self.assertIsNone(args.n_slices)
        self.assertEqual(args.slice_aggregator, "transformer")

    def test_legacy_splits_file_overrides_fold_assignment(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            split_file = Path(temporary_directory) / "splits.json"
            split_file.write_text(
                json.dumps(
                    {
                        "dataset": "cq500",
                        "splits": {
                            "train": ["patient-b", "patient-a"],
                            "val": ["patient-c"],
                            "test": ["patient-d"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                ["--dataset", "cq500", "--splits-file", str(split_file)]
            )

            patient_ids = saved_split_patient_ids(
                args,
                _ToyPatientDataset(
                    ["patient-a", "patient-b", "patient-c", "patient-d"]
                ),
            )

        self.assertEqual(
            patient_ids,
            (["patient-a", "patient-b"], ["patient-c"], ["patient-d"]),
        )

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
