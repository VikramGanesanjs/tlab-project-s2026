"""Loss functions for slice-pair SSL fine-tuning."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

import dinov3.distributed as distributed


def _sinkhorn_knopp(teacher_output: Tensor, temperature: float, iterations: int = 3) -> Tensor:
    """DINO-style Sinkhorn assignments using the active DINO process subgroup."""
    output = teacher_output.float()
    if output.ndim != 2 or output.shape[0] == 0:
        raise ValueError("Sinkhorn expects non-empty [samples, prototypes] logits")
    assignments = torch.exp(output / temperature).t()
    prototypes = assignments.shape[0]

    if dist.is_initialized():
        process_group = distributed.get_process_subgroup()
        global_batch = torch.tensor(
            assignments.shape[1], device=assignments.device, dtype=assignments.dtype
        )
        dist.all_reduce(global_batch, group=process_group)
    else:
        global_batch = torch.tensor(
            assignments.shape[1], device=assignments.device, dtype=assignments.dtype
        )

    total = assignments.sum()
    if dist.is_initialized():
        dist.all_reduce(total, group=process_group)
    assignments /= total.clamp_min(torch.finfo(assignments.dtype).tiny)
    for _ in range(iterations):
        row_sum = assignments.sum(dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(row_sum, group=process_group)
        assignments /= row_sum.clamp_min(torch.finfo(assignments.dtype).tiny)
        assignments /= prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(torch.finfo(assignments.dtype).tiny)
        assignments /= global_batch
    return (assignments * global_batch).t()


def _weighted_ce(student_logits: Tensor, teacher_probs: Tensor, weights: Tensor, student_temp: float) -> Tuple[Tensor, Tensor]:
    log_probs = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    ce = -(teacher_probs.float() * log_probs).sum(dim=-1)
    weights = weights.to(device=ce.device, dtype=ce.dtype)
    return (ce * weights).sum(), weights.sum()


def _uncertainty_weight(teacher_probs: Tensor, gamma: float) -> Tensor:
    probs = teacher_probs.float().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return 1.0 + gamma * entropy


def uwsd_loss(
    *,
    teacher_logits1: Tensor,
    teacher_logits2: Tensor,
    student_logits1: Tensor,
    student_logits2: Tensor,
    local_logits1: Optional[Tensor],
    local_logits2: Optional[Tensor],
    teacher_temp: float,
    student_temp: float,
    gamma: float,
    cross_view_global_loss_weight: float,
) -> Tensor:
    """UWSD multi-crop DINO loss with slice-aware pair routing."""
    teacher_logits = torch.cat((teacher_logits1, teacher_logits2), dim=0)
    teacher_probs = _sinkhorn_knopp(teacher_logits, teacher_temp)
    batch_size = teacher_logits1.shape[0]
    teacher_probs1, teacher_probs2 = teacher_probs.split(batch_size, dim=0)

    terms: list[Tuple[Tensor, Tensor, Tensor]] = []
    weight1 = _uncertainty_weight(teacher_probs1, gamma)
    weight2 = _uncertainty_weight(teacher_probs2, gamma)

    if local_logits1 is not None:
        for logits in local_logits1:
            terms.append((logits, teacher_probs1, weight1))
    if local_logits2 is not None:
        for logits in local_logits2:
            terms.append((logits, teacher_probs2, weight2))

    terms.append((student_logits1, teacher_probs2, cross_view_global_loss_weight * weight2))
    terms.append((student_logits2, teacher_probs1, cross_view_global_loss_weight * weight1))

    numerator: Optional[Tensor] = None
    denominator: Optional[Tensor] = None
    for student_logits, target_probs, weights in terms:
        term_num, term_den = _weighted_ce(student_logits, target_probs, weights, student_temp)
        numerator = term_num if numerator is None else numerator + term_num
        denominator = term_den if denominator is None else denominator + term_den
    assert numerator is not None and denominator is not None
    return numerator / denominator.clamp_min(1e-8)


def croco_ibot_loss(
    *,
    teacher_logits1: Tensor,
    teacher_logits2: Tensor,
    decoded_tokens1: Tensor,
    decoded_tokens2: Tensor,
    masks1: Tensor,
    masks2: Tensor,
    ibot_head: Callable[[Tensor], Tensor],
    teacher_temp: float,
    student_temp: float,
) -> Tensor:
    """Cross-view masked-patch completion from already decoded patch tokens."""
    teacher_selected1 = teacher_logits1[masks1]
    teacher_selected2 = teacher_logits2[masks2]
    teacher_selected = torch.cat((teacher_selected1, teacher_selected2), dim=0)
    teacher_probs = _sinkhorn_knopp(teacher_selected, teacher_temp)
    count1 = teacher_selected1.shape[0]
    teacher_probs1, teacher_probs2 = teacher_probs.split((count1, teacher_selected2.shape[0]), dim=0)

    student_logits1 = ibot_head(decoded_tokens1)
    student_logits2 = ibot_head(decoded_tokens2)

    def patch_ce(student_logits: Tensor, target: Tensor) -> Tensor:
        if student_logits.numel() == 0:
            return student_logits.sum() * 0.0
        return -(target.float() * F.log_softmax(student_logits.float() / student_temp, dim=-1)).sum(-1).mean()

    return 0.5 * (patch_ce(student_logits1, teacher_probs1) + patch_ce(student_logits2, teacher_probs2))
