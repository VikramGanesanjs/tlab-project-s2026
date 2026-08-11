"""Export a plain DINOv3 teacher backbone from a distributed checkpoint.

The training checkpoints used by this repository store LoRA adapters around
the fused attention QKV projection.  This script loads only the teacher
backbone, folds each adapter into its corresponding Q, K, or V rows, removes
the LoRA wrappers, and saves the resulting plain backbone state dict.  The
output is compatible with a regular raw DINOv3/Vision Transformer ``.pth``
file (i.e. its keys are ``blocks.0...``, ``norm...``, etc., with no
``teacher.`` or ``backbone.`` prefix).

With ``--hub-compatible``, the output is additionally padded to the released
``dinov3_vitb16`` hub schema. The SSL checkpoints intentionally omit storage
tokens and QKV bias masks, so storage tokens are taken from the original
released checkpoint and the added bias masks are set to one.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from vit_lora import LoRA  # noqa: E402

LOGGER = logging.getLogger("merge_dcp_lora")
LORA_PARAMETER_NAMES = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
_BLOCK_QKV_RE = re.compile(r"blocks\.(\d+)\.attn\.qkv\.")
_LORA_QKV_RE = re.compile(r"blocks\.(\d+)\.attn\.qkv\.w_[ab]_[qkv]\.weight$")
_VANILLA_QKV_RE = re.compile(r"blocks\.(\d+)\.attn\.qkv\.weight$")


def _checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Select a backbone-bearing state dict from a regular checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping of parameter names to tensors")

    state: object = checkpoint
    for key in ("teacher", "student"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    else:
        candidate = checkpoint.get("model")
        if isinstance(candidate, dict):
            for key in ("teacher", "student"):
                nested = candidate.get(key)
                if isinstance(nested, dict):
                    state = nested
                    break
            else:
                state = candidate

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a usable state dict")
    return {
        str(name): value
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _normalize_backbone_state(
    state: Dict[str, torch.Tensor],
    encoder: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Keep and normalize only parameters understood by ``encoder``."""
    expected = set(encoder.state_dict())
    normalized: Dict[str, torch.Tensor] = {}
    for original_name, value in state.items():
        name = original_name
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "teacher.", "student.", "backbone."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
                    break

        candidates = [name]
        if ".attn.qkv.qkv." in name:
            candidates.append(name.replace(".attn.qkv.qkv.", ".attn.qkv."))
        elif ".attn.qkv." in name:
            candidates.append(name.replace(".attn.qkv.", ".attn.qkv.qkv."))
        for candidate in candidates:
            if candidate in expected:
                normalized[candidate] = value
                break
    return normalized


def _canonical_checkpoint_name(name: str) -> str:
    """Strip common checkpoint containers from a parameter name."""
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "teacher.", "student.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


def _infer_unfrozen_tail(names: Sequence[str]) -> int:
    """Infer the number of final blocks represented by ordinary QKV layers."""
    block_ids = set()
    lora_blocks = set()
    vanilla_blocks = set()
    for original_name in names:
        name = _canonical_checkpoint_name(str(original_name))
        match = _BLOCK_QKV_RE.search(name)
        if match:
            block_ids.add(int(match.group(1)))
        match = _LORA_QKV_RE.search(name)
        if match:
            lora_blocks.add(int(match.group(1)))
        match = _VANILLA_QKV_RE.search(name)
        if match:
            vanilla_blocks.add(int(match.group(1)))

    if not lora_blocks or not block_ids:
        return 0

    tail = 0
    for block_index in range(max(block_ids), -1, -1):
        if block_index in vanilla_blocks and block_index not in lora_blocks:
            tail += 1
        else:
            break
    return tail


def _build_custom_vit_base(
    *,
    repo_dir: Path,
    ssl_architecture: bool,
    with_lora: bool,
    lora_rank: int,
    unfreeze_last_layers: int = 0,
) -> nn.Module:
    """Build the ViT-B variant used by a custom DINOv3 SSL checkpoint."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from dinov3.models.vision_transformer import vit_base

    if ssl_architecture:
        encoder = vit_base(
            patch_size=16,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_dtype="bf16",
            qkv_bias=True,
            layerscale_init=1.0e-05,
            norm_layer="layernorm",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=0,
            mask_k_bias=False,
        )
    else:
        encoder = vit_base(
            patch_size=16,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            qkv_bias=True,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
    if with_lora:
        from vit_lora import add_lora_to_vit

        add_lora_to_vit(encoder, r=lora_rank)
        if not 0 <= unfreeze_last_layers <= len(encoder.blocks):
            raise ValueError(
                "unfreeze_last_layers must be between 0 and the number of ViT blocks; "
                f"got {unfreeze_last_layers}"
            )
        for block in encoder.blocks[-unfreeze_last_layers:] if unfreeze_last_layers else ():
            if not isinstance(block.attn.qkv, LoRA):
                raise TypeError("Expected a LoRA-wrapped QKV while restoring the unfrozen tail")
            block.attn.qkv = block.attn.qkv.qkv
    return encoder


def _resolve_dcp_checkpoint(path: Path) -> Path:
    """Accept either one DCP directory or a DINOv3 ``ckpt`` parent."""
    if (path / ".metadata").is_file():
        return path
    candidates = sorted(
        (child for child in path.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda child: int(child.name),
    )
    if candidates and (candidates[-1] / ".metadata").is_file():
        resolved = candidates[-1]
        LOGGER.info("Checkpoint directory contains multiple DCP steps; using latest: %s", resolved)
        return resolved
    raise ValueError(
        f"No DINOv3 distributed checkpoint metadata found in directory: {path}"
    )


def _load_dcp_backbone(
    encoder: nn.Module,
    checkpoint_dir: Path,
    *,
    source_prefix: str,
) -> int:
    """Load only one backbone prefix from a DINOv3 DCP checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    state = {name: torch.empty_like(value) for name, value in encoder.state_dict().items()}
    destination: Dict[str, object] = {"model": {}}
    cursor: Dict[str, object] = destination["model"]  # type: ignore[assignment]
    prefix_parts = source_prefix.removesuffix(".").split(".")
    for part in prefix_parts[1:]:
        child: Dict[str, object] = {}
        cursor[part] = child
        cursor = child
    cursor.update(state)

    dcp.load(destination, storage_reader=FileSystemReader(checkpoint_dir))
    loaded_state = cursor
    assert isinstance(loaded_state, dict)
    result = encoder.load_state_dict(loaded_state, strict=True)  # type: ignore[arg-type]
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected distributed-backbone load result: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return len(state)


def _is_lora_parameter_name(name: str) -> bool:
    return any(parameter_name in name for parameter_name in LORA_PARAMETER_NAMES)


def load_custom_dinov3_encoder(
    *,
    checkpoint: Path,
    repo_dir: Path,
    device: torch.device,
    encoder_training: str,
    lora_rank: int,
) -> nn.Module:
    """Load a ViT-B backbone from a regular or distributed custom checkpoint."""
    if checkpoint.is_dir():
        checkpoint = _resolve_dcp_checkpoint(checkpoint)
        from torch.distributed.checkpoint.filesystem import FileSystemReader

        metadata = FileSystemReader(checkpoint).read_metadata()
        metadata_keys = set(metadata.state_dict_metadata)
        prefixes = (
            "model.teacher.backbone.",
            "model.student.backbone.",
            "model.backbone.",
        )
        source_prefix = next(
            (prefix for prefix in prefixes if any(key.startswith(prefix) for key in metadata_keys)),
            None,
        )
        if source_prefix is None:
            raise ValueError(
                "Distributed checkpoint has no ViT backbone under the expected "
                f"prefixes: {checkpoint}"
            )
        backbone_keys = [key for key in metadata_keys if key.startswith(source_prefix)]
        has_lora = any(".w_a_" in key or ".w_b_" in key for key in backbone_keys)
        has_storage_tokens = any(key.endswith(".storage_tokens") for key in backbone_keys)
        unfreeze_last_layers = _infer_unfrozen_tail(backbone_keys)
        checkpoint_lora_rank = next(
            (
                int(metadata.state_dict_metadata[key].size[0])
                for key in backbone_keys
                if key.endswith(".w_a_q.weight")
            ),
            lora_rank,
        )
        encoder = _build_custom_vit_base(
            repo_dir=repo_dir,
            ssl_architecture=not has_storage_tokens,
            with_lora=has_lora,
            lora_rank=checkpoint_lora_rank,
            unfreeze_last_layers=unfreeze_last_layers,
        )
        loaded_count = _load_dcp_backbone(
            encoder, checkpoint, source_prefix=source_prefix.removesuffix(".")
        )
        LOGGER.info(
            "Loaded ViT-B backbone from distributed checkpoint %s (%d tensors); "
            "LoRA=%s, unfrozen_tail=%d, rank=%d; discarded decoder, heads, losses, and optimizer",
            checkpoint,
            loaded_count,
            has_lora,
            unfreeze_last_layers,
            checkpoint_lora_rank,
        )
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Custom DINOv3 checkpoint not found: {checkpoint}")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _checkpoint_state_dict(loaded)
        has_lora = any(".w_a_" in key or ".w_b_" in key for key in state)
        unfreeze_last_layers = _infer_unfrozen_tail(state)
        has_storage_tokens = any(
            _canonical_checkpoint_name(key).endswith("storage_tokens") for key in state
        )
        checkpoint_lora_rank = next(
            (int(value.shape[0]) for key, value in state.items() if key.endswith(".w_a_q.weight")),
            lora_rank,
        )
        encoder = _build_custom_vit_base(
            repo_dir=repo_dir,
            ssl_architecture=not has_storage_tokens,
            with_lora=has_lora,
            lora_rank=checkpoint_lora_rank,
            unfreeze_last_layers=unfreeze_last_layers,
        )
        backbone_state = _normalize_backbone_state(state, encoder)
        if not backbone_state:
            raise ValueError(f"No ViT-B backbone parameters found in checkpoint: {checkpoint}")
        incompatible = encoder.load_state_dict(backbone_state, strict=False)
        missing = [key for key in incompatible.missing_keys if key in encoder.state_dict()]
        if missing:
            raise ValueError(
                f"Custom checkpoint is missing {len(missing)} ViT-B backbone tensors; "
                f"first missing keys: {missing[:5]}"
            )
        LOGGER.info(
            "Loaded ViT-B backbone from regular checkpoint %s (%d tensors); "
            "LoRA=%s, unfrozen_tail=%d, rank=%d; discarded non-backbone entries",
            checkpoint,
            len(backbone_state),
            has_lora,
            unfreeze_last_layers,
            checkpoint_lora_rank,
        )

    has_encoder_lora = any(
        _is_lora_parameter_name(name) for name, _ in encoder.named_parameters()
    )
    if encoder_training == "lora" and not has_encoder_lora:
        from vit_lora import add_lora_to_vit

        add_lora_to_vit(encoder, r=lora_rank)
    if encoder_training == "lora":
        from vit_lora import freeze_non_lora_parameters

        freeze_non_lora_parameters(encoder)
        if unfreeze_last_layers:
            encoder.blocks[-unfreeze_last_layers:].requires_grad_(True)
    return encoder.to(device).eval()


def _find_teacher_prefix(metadata_keys: Mapping[str, object]) -> str:
    """Find the teacher backbone prefix used by DINOv3 DCP checkpoints."""
    prefixes = (
        "model.teacher.backbone.",
        "model.student.backbone.",
        "model.backbone.",
    )
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in metadata_keys):
            return prefix
    raise ValueError(
        "Distributed checkpoint has no ViT backbone under the expected "
        f"prefixes: {prefixes}"
    )


@torch.no_grad()
def merge_lora_qkv(model: nn.Module) -> int:
    """Fold all LoRA Q/K/V updates into fused QKV weights in ``model``.

    ``LoRA`` computes ``base(x) + B(A(x))`` for each projection.  Since a
    linear layer applies ``x @ weight.T``, the equivalent weight update is
    ``B.weight @ A.weight``.  The three updates are concatenated in Q/K/V
    order because the base layer is a fused QKV projection.

    Returns the number of LoRA-wrapped attention layers merged.
    """
    merged = 0
    for block_index, block in enumerate(model.blocks):
        qkv = block.attn.qkv
        if not isinstance(qkv, LoRA):
            continue

        base = qkv.qkv
        if not isinstance(base, nn.Linear) or base.bias is None:
            raise TypeError(
                f"blocks.{block_index}.attn.qkv must wrap a biased nn.Linear"
            )
        if base.out_features != 3 * base.in_features:
            raise ValueError(
                f"blocks.{block_index}.attn.qkv is not a fused QKV layer: "
                f"in={base.in_features}, out={base.out_features}"
            )

        updates = torch.cat(
            [
                qkv.w_b_q.weight @ qkv.w_a_q.weight,
                qkv.w_b_k.weight @ qkv.w_a_k.weight,
                qkv.w_b_v.weight @ qkv.w_a_v.weight,
            ],
            dim=0,
        )
        if updates.shape != base.weight.shape:
            raise ValueError(
                f"LoRA update shape {tuple(updates.shape)} does not match "
                f"QKV weight shape {tuple(base.weight.shape)} at block {block_index}"
            )

        base.weight.add_(updates)
        block.attn.qkv = base
        merged += 1

    return merged


def _checkpoint_lora_rank(metadata, backbone_keys: list[str], fallback: int) -> int:
    for key in backbone_keys:
        if key.endswith(".w_a_q.weight"):
            return int(metadata.state_dict_metadata[key].size[0])
    return fallback


def _make_hub_compatible_state(
    trained_state: Mapping[str, torch.Tensor],
    template_path: Path,
) -> tuple[OrderedDict[str, torch.Tensor], int]:
    """Pad an SSL backbone state dict to the released hub model schema."""
    template = torch.load(template_path, map_location="cpu", weights_only=False)
    if not isinstance(template, Mapping):
        raise ValueError(f"Hub template must contain a state dict: {template_path}")

    template_state = OrderedDict(
        (str(name), value.detach().cpu().clone())
        for name, value in template.items()
        if isinstance(value, torch.Tensor)
    )
    missing_from_template = sorted(set(trained_state) - set(template_state))
    if missing_from_template:
        raise ValueError(
            "Trained backbone contains keys absent from the hub template: "
            f"{missing_from_template[:8]}"
        )

    for name, value in trained_state.items():
        if template_state[name].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: trained={tuple(value.shape)}, "
                f"hub_template={tuple(template_state[name].shape)}"
            )
        template_state[name] = value.detach().cpu().clone()

    template_only = set(template_state) - set(trained_state)
    expected_template_only = {
        "storage_tokens",
        *(f"blocks.{i}.attn.qkv.bias_mask" for i in range(12)),
    }
    if template_only != expected_template_only:
        raise ValueError(
            "Unexpected hub-template-only keys; refusing to silently discard or "
            f"invent parameters: {sorted(template_only)}"
        )

    # The SSL model was built with mask_k_bias=False, so its trained K biases
    # are active. All-one masks preserve that behavior in the hub model,
    # whose module type is nevertheless LinearKMaskedBias.
    for name in template_only:
        if name.endswith(".bias_mask"):
            template_state[name] = torch.ones_like(template_state[name])
    return template_state, len(template_only)


def export_teacher_backbone(
    checkpoint: Path,
    output: Path,
    *,
    repo_dir: Path,
    fallback_lora_rank: int = 4,
    hub_compatible: bool = False,
    hub_template: Path | None = None,
) -> dict[str, int | str]:
    """Convert one DCP checkpoint and save its plain teacher backbone."""
    checkpoint = _resolve_dcp_checkpoint(Path(checkpoint))
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    metadata = FileSystemReader(checkpoint).read_metadata()
    source_prefix = _find_teacher_prefix(metadata.state_dict_metadata)
    backbone_keys = [
        key for key in metadata.state_dict_metadata if key.startswith(source_prefix)
    ]
    has_lora = any(".w_a_" in key or ".w_b_" in key for key in backbone_keys)
    has_storage_tokens = any(key.endswith(".storage_tokens") for key in backbone_keys)
    lora_rank = _checkpoint_lora_rank(metadata, backbone_keys, fallback_lora_rank)
    unfrozen_tail = _infer_unfrozen_tail(backbone_keys)

    encoder = _build_custom_vit_base(
        repo_dir=Path(repo_dir),
        ssl_architecture=not has_storage_tokens,
        with_lora=has_lora,
        lora_rank=lora_rank,
        unfreeze_last_layers=unfrozen_tail,
    )
    loaded_count = _load_dcp_backbone(
        encoder,
        checkpoint,
        source_prefix=source_prefix.removesuffix("."),
    )
    merged_count = merge_lora_qkv(encoder)

    state = OrderedDict(
        (name, value.detach().cpu()) for name, value in encoder.state_dict().items()
    )
    hub_padded = 0
    if hub_compatible:
        if hub_template is None:
            hub_template = (
                Path(repo_dir).parent
                / "dinov3-weights"
                / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )
        if not hub_template.is_file():
            raise FileNotFoundError(f"Hub-compatible template not found: {hub_template}")
        state, hub_padded = _make_hub_compatible_state(state, hub_template)
    if any("w_a_" in name or "w_b_" in name or ".qkv.qkv." in name for name in state):
        raise RuntimeError("LoRA parameters or nested QKV names remain after merging")
    if any(name.startswith(("teacher.", "student.", "backbone.")) for name in state):
        raise RuntimeError("Output state dict still contains a model container prefix")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    LOGGER.info(
        "Saved %d backbone tensors to %s (loaded=%d, LoRA layers merged=%d, "
        "rank=%d, unfrozen tail=%d)",
        len(state),
        output,
        loaded_count,
        merged_count,
        lora_rank,
        unfrozen_tail,
    )
    if hub_compatible:
        LOGGER.info(
            "Padded output with %d hub-schema tensors from %s",
            hub_padded,
            hub_template,
        )
    return {
        "loaded_tensors": loaded_count,
        "saved_tensors": len(state),
        "lora_layers_merged": merged_count,
        "lora_rank": lora_rank,
        "unfrozen_tail": unfrozen_tail,
        "hub_padded_tensors": hub_padded,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge teacher-backbone LoRA weights from a DINOv3 DCP checkpoint"
    )
    parser.add_argument("checkpoint", type=Path, help="DCP step directory or its ckpt parent")
    parser.add_argument("output", type=Path, help="Output raw backbone .pth path")
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=_SRC_DIR.parent / "opt" / "dinov3",
        help="Local DINOv3 source checkout used to construct the ViT",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=4,
        help="Fallback rank when the checkpoint metadata has no LoRA tensors",
    )
    parser.add_argument(
        "--hub-compatible",
        action="store_true",
        help=(
            "Pad the SSL backbone with the released DINOv3 storage tokens and "
            "QKV bias-mask buffers so torch.hub.load can load it strictly"
        ),
    )
    parser.add_argument(
        "--hub-template",
        type=Path,
        default=None,
        help="Released raw DINOv3 .pth used for hub-only schema tensors",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    result = export_teacher_backbone(
        args.checkpoint,
        args.output,
        repo_dir=args.dinov3_repo,
        fallback_lora_rank=args.lora_rank,
        hub_compatible=args.hub_compatible,
        hub_template=args.hub_template,
    )
    print(result)


if __name__ == "__main__":
    main()
