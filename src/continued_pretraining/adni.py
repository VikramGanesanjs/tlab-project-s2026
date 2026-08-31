"""Deprecated compatibility shim for legacy ``ADNI:...`` dataset paths.

New continued-pretraining configs use :mod:`continued_pretraining.data`, which
wraps the maintained implementation in :mod:`datasets.adni`.
"""

from __future__ import annotations

from typing import Callable, Optional

from datasets.adni import ADNIClassificationDataset

from .data import TlabExtendedVisionDataset, _identity_pair


class ADNI(TlabExtendedVisionDataset):
    """Legacy name retaining the old DINOv3 dataset-path constructor."""

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        dataset = ADNIClassificationDataset(
            root=root,
            csv_path=extra,
            task="cn_mci_ad",
            transforms=_identity_pair,
            augment=False,
        )
        super().__init__(
            dataset,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )


__all__ = ["ADNI"]
