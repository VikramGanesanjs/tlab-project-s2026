# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This file is derived from dinov3/loss/gram_loss.py and is used under the
# terms of the DINOv3 License Agreement.

"""Gram-matrix distillation loss with optional local spatial windowing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """MSE between feature Gram matrices.

    ``spatial_window_size`` is the side length of an odd square patch
    neighborhood.  For example, ``3`` retains each patch and its 8 immediate
    neighbors.  Entries outside that window are zeroed in *both* Gram
    matrices before the MSE is computed.  ``None`` or ``0`` preserves the
    original DINOv3 behavior.
    """

    def __init__(
        self,
        apply_norm: bool = True,
        img_level: bool = True,
        remove_neg: bool = True,
        remove_only_teacher_neg: bool = False,
        spatial_window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        self.spatial_window_size = self._validate_spatial_window_size(spatial_window_size)

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    @staticmethod
    def _validate_spatial_window_size(value: Optional[int]) -> Optional[int]:
        if value is None or value == 0:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 2 == 0:
            raise ValueError(
                "spatial_window_size must be 0/None (disabled) or a positive odd "
                f"integer, got {value!r}"
            )
        return value

    def _spatial_window_mask(self, n_patches: int, device: torch.device) -> torch.Tensor:
        """Return an ``[N, N]`` mask for a square grid of patch tokens.

        Patch centers are one patch-size apart.  A window of side ``n`` has
        half-width ``(n - 1) / 2`` patches, hence its farthest corner lies at
        ``sqrt(2) * (n - 1) / 2 * patch_size``.  Working in patch units gives
        the equivalent test without needing the image-space patch size.
        """
        if self.spatial_window_size is None:
            return torch.ones((n_patches, n_patches), dtype=torch.bool, device=device)

        grid_size = math.isqrt(n_patches)
        if grid_size * grid_size != n_patches:
            raise ValueError(
                "Spatial Gram windowing requires a square patch grid; got "
                f"{n_patches} patch tokens."
            )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, device=device),
                torch.arange(grid_size, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(n_patches, 2)
        squared_distances = (coordinates[:, None] - coordinates[None, :]).square().sum(dim=-1)
        half_width = (self.spatial_window_size - 1) // 2
        return squared_distances <= 2 * half_width * half_width

    def forward(self, output_feats: torch.Tensor, target_feats: torch.Tensor, img_level: bool = True) -> torch.Tensor:
        """Compute the Gram loss for ``[B, N, D]`` features."""
        if img_level:
            if output_feats.ndim != 3 or target_feats.ndim != 3:
                raise ValueError("Image-level Gram loss expects [batch, patches, channels] tensors")
            if output_feats.shape[:2] != target_feats.shape[:2]:
                raise ValueError(
                    "Student and teacher Gram features must share batch and patch dimensions, got "
                    f"{tuple(output_feats.shape)} and {tuple(target_feats.shape)}"
                )
        elif self.spatial_window_size is not None:
            raise ValueError("Spatial Gram windowing is only defined for image-level Gram matrices")

        output_feats = output_feats.float()
        target_feats = target_feats.float()

        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)
        if not img_level and target_feats.ndim == 3:
            target_feats = target_feats.flatten(0, 1)
        target_sim = target_feats @ target_feats.transpose(-1, -2)

        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)
        if not img_level and output_feats.ndim == 3:
            output_feats = output_feats.flatten(0, 1)
        student_sim = output_feats @ output_feats.transpose(-1, -2)

        if self.remove_neg:
            target_sim = target_sim.clamp_min(0.0)
            student_sim = student_sim.clamp_min(0.0)
        elif self.remove_only_teacher_neg:
            negative_teacher = target_sim < 0
            target_sim = target_sim.masked_fill(negative_teacher, 0.0)
            student_sim = student_sim.masked_fill((student_sim < 0) & negative_teacher, 0.0)

        if self.spatial_window_size is not None:
            mask = self._spatial_window_mask(target_sim.shape[-1], target_sim.device)
            target_sim = target_sim.masked_fill(~mask, 0.0)
            student_sim = student_sim.masked_fill(~mask, 0.0)

        return self.mse_loss(student_sim, target_sim)
