"""Checkpoint shim allowing vanilla DINOv3 weights to omit new LoRA keys."""

from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dcpsd

from dinov3.checkpointer import load_checkpoint


def _extract_state_dict(checkpoint):
    """Accept DINO teacher checkpoints and raw backbone state dictionaries."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    for key in ("teacher", "state_dict", "model", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _translate_key(key, target_keys, lora_enabled):
    key = str(key)
    while key.startswith(("module.", "teacher.", "student.")):
        key = key.split(".", 1)[1]

    candidates = [key]
    if not key.startswith("backbone."):
        candidates.append(f"backbone.{key}")

    if lora_enabled:
        candidates += [
            candidate.replace(f".attn.qkv.{suffix}", f".attn.qkv.qkv.{suffix}")
            for candidate in tuple(candidates)
            for suffix in ("weight", "bias", "bias_mask")
            if candidate.endswith(f".attn.qkv.{suffix}")
        ]

    for candidate in candidates:
        if candidate in target_keys:
            return candidate
    return None


def init_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: List[str] | None = None,
    keys_not_sharded: List[str] | None = None,
    process_group: dist.ProcessGroup = None,
    strict_loading: bool = True,
):
    skip_load_keys = skip_load_keys or []
    keys_not_sharded = keys_not_sharded or []
    if Path(checkpoint_path).is_dir():
        return load_checkpoint(
            ckpt_dir=checkpoint_path,
            model=model,
            process_group=process_group,
            strict_loading=strict_loading,
        )

    checkpoint = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    target_keys = set(model.state_dict())
    normalized_checkpoint = {}
    for key, value in checkpoint.items():
        normalized_key = _translate_key(key, target_keys, lora_enabled=not strict_loading)
        if normalized_key is not None:
            normalized_checkpoint[normalized_key] = value
    if not normalized_checkpoint:
        raise ValueError(f"No compatible backbone weights found in {checkpoint_path}")
    checkpoint = {
        key: value
        for key, value in normalized_checkpoint.items()
        if not any(marker in key for marker in skip_load_keys)
    }
    dcpsd.set_model_state_dict(
        model,
        checkpoint,
        options=dcpsd.StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict_loading,
        ),
    )
