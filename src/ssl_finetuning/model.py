"""
Adapted from DINOv3 SSLMetaArch script
"""

from __future__ import annotations

import gc
import logging
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_DINOV3_DIR = _MODULE_DIR.parents[1] / "opt" / "dinov3"
for _path in (_MODULE_DIR, _SRC_DIR, _DINOV3_DIR):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

import torch
import torch.distributed as dist
from torch import Tensor, nn

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.data import DataAugmentationDINO
from dinov3.data.masking import MaskingGenerator
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, iBOTPatchLoss
from dinov3.models import build_model_from_cfg
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters

from .cvd import CrossViewDecoder, _patch_positions
from .losses import croco_ibot_loss as compute_croco_ibot_loss
from .losses import uwsd_loss as compute_uwsd_loss

logger = logging.getLogger("dinov3")

_LORA_MARKERS = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")


def _add_lora_with_unfrozen_tail(model: nn.Module, rank: int, unfreeze_last_layers: int) -> None:
    """Attach LoRA except for the fully trainable final backbone blocks."""
    from utils.vit_lora import LoRA, add_lora_to_vit

    n_blocks = len(model.blocks)
    if not 0 <= unfreeze_last_layers <= n_blocks:
        raise ValueError(
            f"unfreeze_last_layers must be between 0 and {n_blocks}, got {unfreeze_last_layers}"
        )

    add_lora_to_vit(model, r=rank)
    for block in model.blocks[-unfreeze_last_layers:] if unfreeze_last_layers else ():
        if not isinstance(block.attn.qkv, LoRA):
            raise TypeError("Expected LoRA-wrapped QKV while restoring the unfrozen tail")
        block.attn.qkv = block.attn.qkv.qkv


def _unfreeze_norms_in_lora_blocks(model: nn.Module, unfreeze_last_layers: int) -> int:
    """Make block normalization parameters trainable wherever LoRA is active."""
    lora_blocks = model.blocks[:-unfreeze_last_layers] if unfreeze_last_layers else model.blocks
    unfrozen_parameters = 0
    for block in lora_blocks:
        for module_name, module in block.named_modules():
            if "norm" not in module_name.lower():
                continue
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad_(True)
                    unfrozen_parameters += parameter.numel()
    return unfrozen_parameters


def _unfreeze_backbone_tail(model: nn.Module, unfreeze_last_layers: int) -> None:
    if unfreeze_last_layers:
        model.blocks[-unfreeze_last_layers:].requires_grad_(True)


def _set_patch_embeddings_trainable(model: nn.Module, trainable: bool) -> int:
    """Set the trainability of every parameter in a ViT patch embedder."""
    if not hasattr(model, "patch_embed"):
        raise AttributeError(f"{type(model).__name__} has no patch_embed module")
    model.patch_embed.requires_grad_(trainable)
    return sum(parameter.numel() for parameter in model.patch_embed.parameters())


class SSLFineTune(nn.Module):
    """DINOv3 slice-pair fine-tuning with cross-view masked completion."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Number of backbone parameters: %s", count_parameters(student_backbone))

        # The released backbone is frozen. If LoRA is added by the surrounding
        # setup, its parameters remain trainable; all ordinary backbone weights
        # are excluded from gradient updates and EMA below.
        for parameter in student_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in teacher_backbone.parameters():
            parameter.requires_grad_(False)

        self.embed_dim = int(embed_dim)
        self.dino_out_dim = int(cfg.dino.head_n_prototypes)
        dino_head_class = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=self.dino_out_dim,
            hidden_dim=int(cfg.dino.head_hidden_dim),
            bottleneck_dim=int(cfg.dino.head_bottleneck_dim),
            nlayers=int(cfg.dino.head_nlayers),
        )

        ibot_out_dim = int(cfg.ibot.head_n_prototypes)
        ibot_head_class = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=ibot_out_dim,
            hidden_dim=int(cfg.ibot.head_hidden_dim),
            bottleneck_dim=int(cfg.ibot.head_bottleneck_dim),
            nlayers=int(cfg.ibot.head_nlayers),
        )

        self.student = nn.ModuleDict(
            {
                "backbone": student_backbone,
                "dino_head": dino_head_class(),
                "ibot_head": ibot_head_class(),
                "cvd": CrossViewDecoder(cfg, enc_embed_dim=self.embed_dim),
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": teacher_backbone,
                "dino_head": dino_head_class(),
                "ibot_head": ibot_head_class(),
            }
        )

        self.lora_enabled = bool(cfg.lora.enabled)
        self.lora_rank = int(cfg.lora.rank)
        self.unfreeze_last_layers = int(getattr(cfg, "unfreeze_last_layers", 0))
        self.unfreeze_patch_embeddings = bool(getattr(cfg, "unfreeze_patch_embeddings", False))
        if self.unfreeze_last_layers < 0:
            raise ValueError(f"unfreeze_last_layers must be non-negative, got {self.unfreeze_last_layers}")
        if self.unfreeze_last_layers > len(student_backbone.blocks):
            raise ValueError(
                f"unfreeze_last_layers must be no greater than the number of backbone blocks "
                f"({len(student_backbone.blocks)}), got {self.unfreeze_last_layers}"
            )
        self._lora_attached = False
        # Adapters must exist before FSDP2 wraps the backbone so their
        # parameters are included in the sharded module and optimizer groups.
        if self.lora_enabled:
            from utils.vit_lora import freeze_non_lora_parameters

            _add_lora_with_unfrozen_tail(
                self.student.backbone, self.lora_rank, self.unfreeze_last_layers
            )
            _add_lora_with_unfrozen_tail(
                self.teacher.backbone, self.lora_rank, self.unfreeze_last_layers
            )
            freeze_non_lora_parameters(self.student.backbone)
            freeze_non_lora_parameters(self.teacher.backbone)
            unfrozen_norm_parameters = _unfreeze_norms_in_lora_blocks(
                self.student.backbone, self.unfreeze_last_layers
            )
            _unfreeze_backbone_tail(self.student.backbone, self.unfreeze_last_layers)
            logger.info(
                "LoRA enabled with rank=%d; trainable norm parameters in LoRA blocks=%d; "
                "fully unfrozen student backbone tail blocks=%d",
                self.lora_rank,
                unfrozen_norm_parameters,
                self.unfreeze_last_layers,
            )
            self._lora_attached = True
        elif self.unfreeze_last_layers:
            _unfreeze_backbone_tail(self.student.backbone, self.unfreeze_last_layers)

        self.teacher.requires_grad_(False)
        self.model_ema = self.teacher
        student_patch_embed_parameters = _set_patch_embeddings_trainable(
            self.student.backbone, self.unfreeze_patch_embeddings
        )
        teacher_patch_embed_parameters = _set_patch_embeddings_trainable(
            self.teacher.backbone, self.unfreeze_patch_embeddings
        )
        logger.info(
            "Patch embeddings trainable=%s; student parameters=%d; teacher parameters=%d",
            self.unfreeze_patch_embeddings,
            student_patch_embed_parameters,
            teacher_patch_embed_parameters,
        )

        self.dino_loss = DINOLoss(self.dino_out_dim)
        self.ibot_patch_loss = iBOTPatchLoss(ibot_out_dim)
        self._distributed_prepared = False
        self.uwsd_loss_weight = float(cfg.uwsd_loss_weight)
        self.croco_ibot_loss_weight = float(cfg.croco_ibot_loss_weight)
        self.cross_view_global_loss_weight = float(cfg.cross_view_global_loss_weight)
        self.gamma = float(cfg.gamma)
        self.teacher_temp = float(cfg.teacher.teacher_temp)
        self.mask_ratio_min, self.mask_ratio_max = tuple(cfg.ibot.mask_ratio_min_max)
        self.mask_sample_probability = float(cfg.ibot.mask_sample_probability)
        self.ema_params_lists: Optional[Tuple[list[nn.Parameter], list[nn.Parameter]]] = None

        logger.info(
            "Built slice-pair fine-tuner: embed_dim=%d, dino_prototypes=%d, "
            "ibot_prototypes=%d, gamma=%.3f, cross_view_global_loss_weight=%.3f",
            self.embed_dim,
            self.dino_out_dim,
            ibot_out_dim,
            self.gamma,
            self.cross_view_global_loss_weight,
        )

    @property
    def cross_view_decoder(self) -> nn.Module:
        return self.student["cvd"]

    def _copy_student_to_teacher(self) -> None:
        student_state = self.student.state_dict()
        teacher_state = self.teacher.state_dict()
        missing = sorted(name for name in teacher_state if name not in student_state)
        if missing:
            raise RuntimeError(
                "Teacher has parameters/buffers missing from the student state: "
                + ", ".join(missing[:8])
            )
        if self.lora_enabled:
            student_lora_names = {
                name for name in student_state if any(marker in name for marker in _LORA_MARKERS)
            }
            teacher_lora_names = {
                name for name in teacher_state if any(marker in name for marker in _LORA_MARKERS)
            }
            if student_lora_names != teacher_lora_names:
                raise RuntimeError(
                    "Student and teacher LoRA state names do not match: "
                    f"student_only={sorted(student_lora_names - teacher_lora_names)[:8]}, "
                    f"teacher_only={sorted(teacher_lora_names - student_lora_names)[:8]}"
                )
        teacher_state = {name: student_state[name] for name in teacher_state}
        self.teacher.load_state_dict(teacher_state, strict=True)

    def _load_pretrained_backbone(self, checkpoint: str) -> None:
        """Load either a raw DINOv3 backbone file or a training checkpoint."""
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            init_fsdp_model_from_checkpoint(
                self.student,
                checkpoint,
                skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )
            return

        loaded = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(loaded, dict) and "teacher" in loaded:
            loaded = loaded["teacher"]
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise ValueError(f"Unsupported pretrained checkpoint format: {checkpoint}")

        backbone_state = {}
        expected_backbone_keys = {
            name[len("backbone.") :]
            for name in self.student.state_dict()
            if name.startswith("backbone.")
        }
        for name, value in loaded.items():
            name = str(name)
            for prefix in ("module.", "teacher.", "student.", "backbone."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            candidates = [name]
            if self._lora_attached:
                # LoRA blocks store the released fused projection under
                # ``attn.qkv.qkv``. Fully unfrozen tail blocks retain the
                # original ``attn.qkv`` name, so choose whichever key exists.
                candidates.extend(
                    name.replace(f".attn.qkv.{suffix}", f".attn.qkv.qkv.{suffix}")
                    for suffix in ("weight", "bias", "bias_mask")
                    if name.endswith(f".attn.qkv.{suffix}")
                )
            matching_name = next(
                (candidate for candidate in candidates if candidate in expected_backbone_keys),
                None,
            )
            if matching_name is not None:
                backbone_state[matching_name] = value

        if not backbone_state:
            raise ValueError(f"No backbone parameters found in pretrained checkpoint: {checkpoint}")
        if self._distributed_prepared:
            # FSDP2 parameters are DTensors after ``prepare_for_distributed_training``.
            # Match DINOv3's checkpoint loader by converting sharded backbone
            # tensors to the active process mesh before loading them.
            from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
            from torch.distributed.tensor import Shard, distribute_tensor

            process_group = distributed.get_process_subgroup()
            if process_group is None:
                world_mesh = init_device_mesh(
                    "cuda",
                    mesh_shape=(dist.get_world_size(),),
                    mesh_dim_names=("dp",),
                )
            else:
                world_mesh = DeviceMesh.from_group(process_group, "cuda")

            keys_not_sharded = ("backbone.rope_embed.periods", "qkv.bias_mask")
            student_state = {
                f"backbone.{name}": (
                    value
                    if any(key in f"backbone.{name}" for key in keys_not_sharded)
                    # ``distribute_tensor`` defaults to Replicate. FSDP2
                    # parameters are Shard(dim=0), so make that placement
                    # explicit to avoid a replicated-to-sharded copy_().
                    else distribute_tensor(
                        value,
                        world_mesh,
                        placements=[Shard(0)] if value.ndim > 0 else None,
                        src_data_rank=None,
                    )
                )
                for name, value in backbone_state.items()
            }
        else:
            student_state = {f"backbone.{name}": value for name, value in backbone_state.items()}

        # Load through the FSDP-wrapped student container. Loading directly
        # into ``student.backbone`` bypasses FSDP2's state-dict placement
        # handling for replicated parameters such as cls_token and mask_token.
        missing, unexpected = self.student.load_state_dict(student_state, strict=False)
        logger.info(
            "Loaded pretrained backbone from %s (%d tensors); missing=%d unexpected=%d",
            checkpoint,
            len(backbone_state),
            len(missing),
            len(unexpected),
        )

    def init_weights(self) -> None:
        """Initialize new heads/decoder and copy them into the teacher.

        As in the upstream DINOv3 training code, a pretrained student backbone
        can then be loaded with ``resume_from_teacher_chkpt``. The teacher is
        copied only after that load so both branches start identically.
        """
        self.student.backbone.init_weights()
        if self.lora_enabled:
            from utils.vit_lora import init_lora_parameters

            init_lora_parameters(self.student.backbone)
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.student.cvd.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()

        if self.cfg.student.resume_from_teacher_chkpt:
            checkpoint = self.cfg.student.resume_from_teacher_chkpt
            logger.info("Loading pretrained fine-tuning checkpoint from %s", checkpoint)
            self._load_pretrained_backbone(checkpoint)

        self._copy_student_to_teacher()

    @torch.no_grad()
    def assert_finite_state(self, stage: str) -> None:
        """Raise with parameter names if initialization left non-finite values."""
        non_finite = []
        checked = 0

        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            # FSDP2 exposes sharded parameters as DTensors. Check each local
            # shard; this avoids a full-tensor collective just for validation.
            local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            if local_tensor.is_meta:
                non_finite.append(f"{name} (still on meta device)")
                continue
            if not local_tensor.is_floating_point() and not local_tensor.is_complex():
                continue

            checked += 1
            if not torch.isfinite(local_tensor).all().item():
                finite = torch.isfinite(local_tensor)
                bad_values = int((~finite).sum().item())
                non_finite.append(f"{name} ({bad_values} non-finite values)")

        if non_finite:
            details = "\n".join(f"  - {name}" for name in non_finite[:50])
            if len(non_finite) > 50:
                details += f"\n  - ... and {len(non_finite) - 50} more"
            raise RuntimeError(
                f"Non-finite model state detected {stage}; checked {checked} tensors:\n{details}"
            )
        logger.info("Finite model-state audit passed %s (%d floating tensors checked)", stage, checked)

    @staticmethod
    def _unpack_pair(data: Any) -> Tuple[Tensor, Tensor, Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        """Normalize tuple/dict/DataLoader outputs into the pair representation."""
        local1 = local2 = masks1 = masks2 = None
        if isinstance(data, dict):
            if "slices1" in data and "slices2" in data:
                first, second = data["slices1"], data["slices2"]
            elif "global_crops" in data:
                crops = data["global_crops"]
                if isinstance(crops, (list, tuple)) and len(crops) == 2:
                    first, second = crops
                else:
                    first, second = crops.chunk(2, dim=0)
            elif "collated_global_crops" in data:
                first, second = data["collated_global_crops"].chunk(2, dim=0)
            else:
                raise KeyError("Expected slices1/slices2 or global_crops in pair batch")
            local1 = data.get("local_crops1")
            local2 = data.get("local_crops2")
            if local1 is None and local2 is None and "local_crops" in data:
                local_crops = data["local_crops"]
                if isinstance(local_crops, Tensor):
                    local_count = local_crops.shape[0] // first.shape[0]
                    local_crops = local_crops.unflatten(0, (local_count, first.shape[0]))
                else:
                    local_count = len(local_crops)
                split = local_count // 2
                local1 = local_crops[:split]
                local2 = local_crops[split:]
            if local1 is None and "collated_local_crops" in data:
                collated_local = data["collated_local_crops"]
                local_count = collated_local.shape[0] // first.shape[0]
                local_crops = collated_local.unflatten(0, (local_count, first.shape[0]))
                split = max(1, local_count // 2)
                local1 = local_crops[:split]
                local2 = local_crops[split:] if split < local_count else None
            masks1 = data.get("masks1")
            masks2 = data.get("masks2")
            if masks1 is None and "collated_masks" in data:
                masks1, masks2 = data["collated_masks"].chunk(2, dim=0)
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            first, second = data
        else:
            raise TypeError("Expected a pair tuple/list or pair batch dictionary")
        if not isinstance(first, Tensor) or not isinstance(second, Tensor):
            raise TypeError("Both paired slices must be torch tensors")
        if first.ndim != 4 or second.ndim != 4:
            raise ValueError(
                "Paired slices must be [B, 3, H, W], got "
                f"{tuple(first.shape)} and {tuple(second.shape)}"
            )
        if first.shape[0] != second.shape[0]:
            raise ValueError("Both paired slice batches must have the same batch size")
        return first, second, local1, local2, masks1, masks2

    @staticmethod
    def _normalize_local_crops(crops: Any) -> Optional[Tensor]:
        if crops is None:
            return None
        if isinstance(crops, (tuple, list)):
            if not crops:
                return None
            crops = torch.stack(list(crops), dim=0)
        if crops.ndim == 4:
            crops = crops.unsqueeze(0)
        if crops.ndim != 5:
            raise ValueError(f"Local crops must be [L, B, C, H, W], got {tuple(crops.shape)}")
        return crops

    @torch.no_grad()
    def get_teacher_output(
        self,
        images: Tensor,
    ) -> Dict[str, Tensor]:
        """DINOv3-style teacher forward, extended with full patch logits.

        ``images`` follows the original script convention: ``[n_crops, B,
        C, H, W]``. Teacher images are always clean; masking is only used to
        select iBOT targets after the clean forward.
        """
        n_crops, batch_size, _, _, _ = images.shape
        flat_images = images.flatten(0, 1)
        backbone_out = self.teacher.backbone(flat_images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]
        reg = backbone_out["x_storage_tokens"]
        patches = backbone_out["x_norm_patchtokens"]
        cls_logits = self.teacher.dino_head(cls)
        ibot_logits = self.teacher.ibot_head(patches)

        return {
            "cls_pre_head": cls.unflatten(0, (n_crops, batch_size)),
            "reg_pre_head": reg.unflatten(0, (n_crops, batch_size)),
            "patch_pre_head": patches.unflatten(0, (n_crops, batch_size)),
            "cls_after_head": cls_logits.unflatten(0, (n_crops, batch_size)),
            "ibot_after_head": ibot_logits.unflatten(0, (n_crops, batch_size)),
        }

    def get_student_output(
        self,
        *,
        global_crops: Tensor,
        local_crops: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """DINOv3-style student forward with masked global patch tokens."""
        n_global_crops, batch_size, _, _, _ = global_crops.shape
        flat_global = global_crops.flatten(0, 1)

        if local_crops is not None and local_crops.numel() > 0:
            n_local_crops = local_crops.shape[0]
            flat_local = local_crops.flatten(0, 1)
            global_out, local_out = self.student.backbone(
                [flat_global, flat_local],
                masks=[masks, None],
                is_training=True,
            )
        else:
            n_local_crops = 0
            global_out = self.student.backbone(flat_global, masks=masks, is_training=True)
            local_out = None

        g_cls = global_out["x_norm_clstoken"]
        g_reg = global_out["x_storage_tokens"]
        g_patch = global_out["x_norm_patchtokens"]
        global_cls_logits = self.student.dino_head(g_cls)

        student_global = {
            "cls_pre_head": g_cls.unflatten(0, (n_global_crops, batch_size)),
            "reg_pre_head": g_reg.unflatten(0, (n_global_crops, batch_size)),
            "patch_pre_head": g_patch.unflatten(0, (n_global_crops, batch_size)),
            "cls_after_head": global_cls_logits.unflatten(0, (n_global_crops, batch_size)),
        }

        if local_out is None:
            local_cls = g_cls.new_empty((0, batch_size, g_cls.shape[-1]))
            local_reg = g_reg.new_empty((0, batch_size, g_reg.shape[1], g_reg.shape[2]))
            local_patch = g_patch.new_empty((0, batch_size, g_patch.shape[1], g_patch.shape[2]))
            local_logits = global_cls_logits.new_empty((0, batch_size, global_cls_logits.shape[-1]))
        else:
            l_cls = local_out["x_norm_clstoken"]
            l_reg = local_out["x_storage_tokens"]
            l_patch = local_out["x_norm_patchtokens"]
            local_cls = l_cls.unflatten(0, (n_local_crops, batch_size))
            local_reg = l_reg.unflatten(0, (n_local_crops, batch_size))
            local_patch = l_patch.unflatten(0, (n_local_crops, batch_size))
            local_logits = self.student.dino_head(l_cls).unflatten(0, (n_local_crops, batch_size))

        student_local = {
            "cls_pre_head": local_cls,
            "reg_pre_head": local_reg,
            "patch_pre_head": local_patch,
            "cls_after_head": local_logits,
        }
        return student_global, student_local

    def _sample_masks(self, batch_size: int, num_patches: int, device: torch.device) -> Tensor:
        grid_h = int(math.sqrt(num_patches))
        while grid_h > 1 and num_patches % grid_h:
            grid_h -= 1
        grid_w = num_patches // grid_h
        generator = MaskingGenerator((grid_h, grid_w), max_num_patches=num_patches)
        masks = []
        for _ in range(batch_size):
            if torch.rand((), device=device).item() > self.mask_sample_probability:
                mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
            else:
                ratio = torch.empty((), device=device).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()
                count = max(1, min(num_patches, int(round(num_patches * ratio))))
                mask = torch.as_tensor(generator(count), dtype=torch.bool, device=device).flatten()
            masks.append(mask)
        result = torch.stack(masks, dim=0)
        if not result.any():
            result[0, 0] = True
        return result

    def _prepare_mask(
        self,
        masks: Optional[Tensor],
        batch_size: int,
        num_patches: int,
        device: torch.device,
    ) -> Tensor:
        if masks is None:
            return self._sample_masks(batch_size, num_patches, device)
        masks = masks.to(device=device, dtype=torch.bool)
        if masks.shape != (batch_size, num_patches):
            raise ValueError(
                f"Expected masks [{batch_size}, {num_patches}], got {tuple(masks.shape)}"
            )
        if not masks.any():
            masks = masks.clone()
            masks[0, 0] = True
        return masks

    def _decode_masked_queries(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        masks: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Decode masked queries while preserving each patch's source position.

        Masks can contain different numbers of patches for different samples,
        so each sample is decoded separately. This avoids padding tokens or
        allowing attention to mix unrelated samples in the batch.
        """
        decoded = []
        for batch_index in range(query_tokens.shape[0]):
            sample_mask = masks[batch_index].bool()
            if not sample_mask.any():
                continue

            query = query_tokens[batch_index : batch_index + 1, sample_mask]
            query_pos = positions[batch_index : batch_index + 1, sample_mask]
            if self.cross_view_decoder.context_mode == "masked":
                context = context_tokens[batch_index : batch_index + 1, sample_mask]
                context_pos = positions[batch_index : batch_index + 1, sample_mask]
            else:
                context = context_tokens[batch_index : batch_index + 1]
                context_pos = positions[batch_index : batch_index + 1]

            decoded.append(
                self.cross_view_decoder(
                    query,
                    context.detach(),
                    query_pos=query_pos,
                    context_pos=context_pos,
                ).squeeze(0)
            )

        if not decoded:
            return query_tokens.new_empty((0, query_tokens.shape[-1]))
        return torch.cat(decoded, dim=0)

    def compute_losses(
        self,
        *,
        teacher_global: Dict[str, Tensor],
        student_global: Dict[str, Tensor],
        student_local: Dict[str, Tensor],
        masks: Tensor,
        teacher_temp: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the two slice-pair losses."""

        teacher1 = {
            "cls_logits": teacher_global["cls_after_head"][0],
            "patch_pre_head": teacher_global["patch_pre_head"][0],
            "ibot_logits": teacher_global["ibot_after_head"][0],
        }
        teacher2 = {
            "cls_logits": teacher_global["cls_after_head"][1],
            "patch_pre_head": teacher_global["patch_pre_head"][1],
            "ibot_logits": teacher_global["ibot_after_head"][1],
        }
        student1 = {
            "cls_logits": student_global["cls_after_head"][0],
            "patch_pre_head": student_global["patch_pre_head"][0],
        }
        student2 = {
            "cls_logits": student_global["cls_after_head"][1],
            "patch_pre_head": student_global["patch_pre_head"][1],
        }

        local_logits1 = local_logits2 = None
        local_logits = student_local.get("cls_after_head")
        if local_logits is not None and local_logits.shape[0] > 0:
            split = max(1, local_logits.shape[0] // 2)
            local_logits1 = local_logits[:split]
            local_logits2 = local_logits[split:]
            if local_logits2.shape[0] == 0:
                local_logits2 = None

        effective_teacher_temp = float(teacher_temp or self.teacher_temp)
        uwsd_loss = compute_uwsd_loss(
            teacher_logits1=teacher1["cls_logits"],
            teacher_logits2=teacher2["cls_logits"],
            student_logits1=student1["cls_logits"],
            student_logits2=student2["cls_logits"],
            local_logits1=local_logits1,
            local_logits2=local_logits2,
            teacher_temp=effective_teacher_temp,
            student_temp=float(getattr(self.dino_loss, "student_temp", 0.1)),
            gamma=self.gamma,
            cross_view_global_loss_weight=self.cross_view_global_loss_weight,
        )
        batch_size = student_global["cls_after_head"].shape[1]
        masks = masks.reshape(2, batch_size, -1)
        _, num_patches, _ = teacher1["patch_pre_head"].shape
        positions = _patch_positions(batch_size, num_patches, teacher1["patch_pre_head"].device)
        decoded_tokens1 = self._decode_masked_queries(
            student1["patch_pre_head"], teacher2["patch_pre_head"], masks[0], positions
        )
        decoded_tokens2 = self._decode_masked_queries(
            student2["patch_pre_head"], teacher1["patch_pre_head"], masks[1], positions
        )
        croco_ibot_loss = compute_croco_ibot_loss(
            teacher_logits1=teacher1["ibot_logits"],
            teacher_logits2=teacher2["ibot_logits"],
            decoded_tokens1=decoded_tokens1,
            decoded_tokens2=decoded_tokens2,
            masks1=masks[0],
            masks2=masks[1],
            ibot_head=self.student.ibot_head,
            teacher_temp=effective_teacher_temp,
            student_temp=float(getattr(self.ibot_patch_loss, "student_temp", 0.1)),
        )
        total = self.uwsd_loss_weight * uwsd_loss + self.croco_ibot_loss_weight * croco_ibot_loss
        return total, {
            "uwsd_loss": uwsd_loss,
            "croco_ibot_loss": croco_ibot_loss,
        }

    def forward_pair(self, data: Any, *, teacher_temp: Optional[float] = None) -> Dict[str, Tensor]:
        slices1, slices2, local1, local2, masks1, masks2 = self._unpack_pair(data)
        device = next(self.student.parameters()).device
        slices1 = slices1.to(device=device, non_blocking=True)
        slices2 = slices2.to(device=device, non_blocking=True)

        global_crops = torch.stack((slices1, slices2), dim=0)
        teacher_global = self.get_teacher_output(global_crops)
        num_patches = teacher_global["patch_pre_head"].shape[2]
        masks1 = self._prepare_mask(masks1, slices1.shape[0], num_patches, device)
        masks2 = self._prepare_mask(masks2, slices2.shape[0], num_patches, device)
        masks = torch.cat((masks1, masks2), dim=0)

        local1_tensor = self._normalize_local_crops(local1)
        local2_tensor = self._normalize_local_crops(local2)
        if local1_tensor is not None:
            local1_tensor = local1_tensor.to(device=device, non_blocking=True)
        if local2_tensor is not None:
            local2_tensor = local2_tensor.to(device=device, non_blocking=True)

        if local1_tensor is not None and local2_tensor is not None:
            local_crops = torch.cat((local1_tensor, local2_tensor), dim=0)
        elif local1_tensor is not None:
            local_crops = local1_tensor
        else:
            local_crops = local2_tensor

        student_global, student_local = self.get_student_output(
            global_crops=global_crops,
            local_crops=local_crops,
            masks=masks,
        )

        total_loss, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            masks=masks,
            teacher_temp=teacher_temp,
        )
        return {
            "loss": total_loss,
            "uwsd_loss": loss_dict["uwsd_loss"],
            "croco_ibot_loss": loss_dict["croco_ibot_loss"],
            "masks1": masks1,
            "masks2": masks2,
        }

    def forward(self, inputs: Any, *, teacher_temp: Optional[float] = None) -> Dict[str, Tensor]:
        return self.forward_pair(inputs, teacher_temp=teacher_temp)

    def forward_backward(
        self, data: Any, *, teacher_temp: Optional[float] = None, iteration: int = 0, **_: Any
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del iteration
        outputs = self.forward_pair(data, teacher_temp=teacher_temp)
        self.backprop_loss(outputs["loss"])
        metrics = {
            "loss": outputs["loss"].detach(),
            "uwsd_loss": outputs["uwsd_loss"].detach(),
            "croco_ibot_loss": outputs["croco_ibot_loss"].detach(),
        }
        return outputs["loss"], metrics

    def train(self, mode: bool = True) -> "SSLFineTune":
        super().train(mode)
        self.teacher.eval()
        return self

    def backprop_loss(self, loss: Tensor) -> None:
        loss.backward()

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        """EMA trainable student heads/backbone parameters using stable names.

        Frozen base-backbone parameters remain excluded. This includes LoRA
        parameters, trainable norms in LoRA blocks, and all parameters in an
        optionally fully unfrozen backbone tail. Name matching is required
        because inserting LoRA wrappers changes the module structure and
        makes positional parameter pairing unsafe.
        """
        student_named = dict(self.student.named_parameters())
        teacher_named = dict(self.teacher.named_parameters())
        if self.ema_params_lists is None:
            student_params: list[nn.Parameter] = []
            teacher_params: list[nn.Parameter] = []
            for name, student_param in student_named.items():
                is_head = name.startswith(("dino_head.", "ibot_head."))
                is_lora = any(marker in name for marker in _LORA_MARKERS)
                is_trainable_backbone = name.startswith("backbone.") and (
                    student_param.requires_grad or is_lora
                )
                if not (is_head or is_trainable_backbone):
                    continue
                teacher_param = teacher_named.get(name)
                if teacher_param is None:
                    raise RuntimeError(
                        "EMA teacher is missing student parameter: " + name
                    )
                student_params.append(student_param)
                teacher_params.append(teacher_param)

            if self.lora_enabled:
                student_lora_names = {
                    name for name in student_named if any(marker in name for marker in _LORA_MARKERS)
                }
                teacher_lora_names = {
                    name for name in teacher_named if any(marker in name for marker in _LORA_MARKERS)
                }
                missing_lora = sorted(student_lora_names - teacher_lora_names)
                extra_lora = sorted(teacher_lora_names - student_lora_names)
                if missing_lora or extra_lora:
                    raise RuntimeError(
                        "Student and teacher LoRA parameters do not match: "
                        f"missing_in_teacher={missing_lora[:8]}, "
                        f"extra_in_teacher={extra_lora[:8]}"
                    )
            self.ema_params_lists = (student_params, teacher_params)

        student_params, teacher_params = self.ema_params_lists
        for teacher_param, student_param in zip(teacher_params, student_params):
            teacher_param.mul_(momentum).add_(student_param, alpha=1.0 - momentum)

    def is_lora_warmup_backbone_parameter(self, name: str, parameter: nn.Parameter) -> bool:
        """Whether a student-backbone parameter follows the delayed LoRA schedule."""
        if not self.lora_enabled:
            return False
        if any(marker in name for marker in _LORA_MARKERS):
            return True
        if self.unfreeze_patch_embeddings and name.startswith("patch_embed."):
            return True
        parts = name.split(".", 2)
        if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
            return False
        lora_block_count = len(self.student.backbone.blocks) - self.unfreeze_last_layers
        return (
            int(parts[1]) < lora_block_count
            and "norm" in name.lower()
            and parameter.requires_grad
        )

    def build_data_augmentation_dino(self, cfg: Any) -> DataAugmentationDINO:
        crops = cfg.crops
        return DataAugmentationDINO(
            crops.global_crops_scale,
            crops.local_crops_scale,
            crops.local_crops_number,
            global_crops_size=crops.global_crops_size,
            local_crops_size=crops.local_crops_size,
            gram_teacher_crops_size=None,
            local_crops_subset_of_global_crops=crops.localcrops_subset_of_globalcrops,
            share_color_jitter=crops.share_color_jitter,
            horizontal_flips=crops.horizontal_flips,
        )

    def get_maybe_fused_params_for_submodel(self, module: nn.Module):
        optim_cfg = self.cfg.optim
        params_groups = get_params_groups_with_decay_fsdp(
            model=module,
            lr_decay_rate=float(optim_cfg.layerwise_decay),
            patch_embed_lr_mult=float(optim_cfg.patch_embed_lr_mult),
            dino_head_wd_multiplier=float(optim_cfg.dino_head_wd_multiplier),
        )
        lora_markers = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
        lora_param_ids = {
            id(parameter)
            for name, parameter in module.named_parameters()
            if any(marker in name for marker in lora_markers)
        }
        for group in params_groups:
            # ``get_params_groups_with_decay_fsdp`` creates one group per
            # parameter.  Fused groups become lists only after this tagging
            # step, so inspect the parameter directly here.
            group["is_lora"] = id(group["params"]) in lora_param_ids
            group["is_lora_warmup"] = (
                module is self.student.backbone
                and self.is_lora_warmup_backbone_parameter(group["name"], group["params"])
            )
        if bool(optim_cfg.multi_tensor_optim):
            fused_groups = fuse_params_groups(
                params_groups,
                keys=(
                    "lr_multiplier",
                    "wd_multiplier",
                    "is_last_layer",
                    "is_lora",
                    "is_lora_warmup",
                ),
            )
            for group in fused_groups:
                group["foreach"] = True
                group["fused"] = True
            return fused_groups
        return params_groups

    def get_params_groups(self):
        return [
            group
            for module in self.student.values()
            for group in self.get_maybe_fused_params_for_submodel(module)
        ]

    def prepare_for_distributed_training(self) -> None:
        process_group = distributed.get_process_subgroup()
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=[self.model_ema],
            cfg=self.cfg,
            trained_model_process_group=process_group,
            inference_only_models_process_groups=[process_group],
        )
        self._distributed_prepared = True

    def broadcast_to_subgroups(self, tensor: Tensor, over_dim: int, global_batch_size: Optional[int] = None) -> Tensor:
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        concatenated = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            concatenated = concatenated.narrow(dim=over_dim, start=0, length=global_batch_size)
        return concatenated.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
