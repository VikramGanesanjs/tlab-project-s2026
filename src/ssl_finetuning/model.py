"""Slice-pair self-supervised fine-tuning for DINOv3.

The implementation keeps the DINOv3 backbone/head conventions used by the
local repository, but changes the training graph to operate on two nearby
slices from the same volume:

* clean teacher features are computed for both slices;
* masked student features are computed for both slices;
* DINO loss compares within-slice local views (when supplied) and the two
  global views cross-slice;
* the cross-view decoder uses the other slice as context for iBOT completion.
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
for _path in (_SRC_DIR, _MODULE_DIR, _DINOV3_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

from utils import DecoderBlock

logger = logging.getLogger("dinov3")


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _DecoderRoPE(nn.Module):
    """Axial 2-D rotary position encoding for the CroCo-style decoder.

    ``DecoderBlock`` calls the rotary module with tensors shaped
    ``[B, heads, tokens, head_dim]`` and integer patch coordinates shaped
    ``[B, tokens, 2]``. The coordinate layout mirrors DINOv3's axial RoPE:
    half of the frequencies encode height and half encode width.
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("decoder embed_dim must be divisible by decoder n_heads")
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("decoder head dimension must be divisible by 4 for 2-D RoPE")
        self.head_dim = head_dim
        self.base = float(base)
        periods = base ** (
            2
            * torch.arange(head_dim // 4, dtype=torch.float32)
            / (head_dim // 2)
        )
        self.register_buffer("periods", periods, persistent=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        periods = self.base ** (
            2
            * torch.arange(
                self.head_dim // 4,
                device=self.periods.device,
                dtype=self.periods.dtype,
            )
            / (self.head_dim // 2)
        )
        self.periods.copy_(periods)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and x.shape[0] != 1:
            positions = positions.expand(x.shape[0], -1, -1)
        if positions.shape[0] != x.shape[0] or positions.shape[1] != x.shape[2]:
            raise ValueError(
                "RoPE positions must match decoder input: "
                f"x={tuple(x.shape)}, positions={tuple(positions.shape)}"
            )

        pos = positions.to(device=x.device, dtype=self.periods.dtype)
        periods = self.periods.to(device=x.device)
        angles = 2.0 * math.pi * pos[..., :, None] / periods[None, None, None, :]
        # [B, N, 2, D/4] -> [B, N, D/2], then duplicate for rotate_half.
        angles = angles.flatten(-2, -1)
        angles = torch.cat((angles, angles), dim=-1)
        sin = angles.sin().unsqueeze(1)
        cos = angles.cos().unsqueeze(1)
        dtype = x.dtype
        x_float = x.float()
        rotated = (x_float * cos) + (_rotate_half(x_float) * sin)
        return rotated.to(dtype=dtype)


def _patch_positions(batch_size: int, num_patches: int, device: torch.device) -> Tensor:
    """Return row-major ``[B, num_patches, 2]`` patch coordinates."""
    height = int(math.sqrt(num_patches))
    while height > 1 and num_patches % height:
        height -= 1
    width = num_patches // height
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    positions = torch.stack((row_grid.flatten(), col_grid.flatten()), dim=-1)
    return positions.unsqueeze(0).expand(batch_size, -1, -1)


class CrossViewDecoder(nn.Module):
    """CroCo-style decoder for cross-slice masked-patch completion."""

    def __init__(self, cfg: Any, enc_embed_dim: Optional[int] = None) -> None:
        super().__init__()
        self.n_blocks = int(cfg.decoder.n_blocks)
        self.embed_dim = int(cfg.decoder.embed_dim or enc_embed_dim)
        self.enc_embed_dim = int(enc_embed_dim)
        self.n_heads = int(cfg.decoder.n_heads)
        self.mlp_ratio = float(cfg.decoder.mlp_ratio)
        rope_base = float(cfg.decoder.rope_base)
        self.context_mode = str(cfg.decoder.context_mode).lower()
        if self.context_mode not in {"masked", "full"}:
            raise ValueError(
                "decoder.context_mode must be 'masked' or 'full', "
                f"got {self.context_mode!r}"
            )

        self.decoder_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
        self.context_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
        self.rope = _DecoderRoPE(self.embed_dim, self.n_heads, base=rope_base)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    self.embed_dim,
                    self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    rope=self.rope,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.dec_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = (
            nn.Identity()
            if self.embed_dim == self.enc_embed_dim
            else nn.Linear(self.embed_dim, self.enc_embed_dim)
        )

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the decoder after meta-device materialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.rope.reset_parameters()

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_pos: Optional[Tensor] = None,
        context_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if query_tokens.ndim != 3 or context_tokens.ndim != 3:
            raise ValueError("CVD expects query/context tokens shaped [B, patches, dim]")
        if query_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "CVD query/context batches must align: "
                f"query={tuple(query_tokens.shape)}, context={tuple(context_tokens.shape)}"
            )
        batch_size, query_patches, _ = query_tokens.shape
        context_patches = context_tokens.shape[1]
        if query_pos is None:
            query_pos = _patch_positions(batch_size, query_patches, query_tokens.device)
        if context_pos is None:
            context_pos = _patch_positions(batch_size, context_patches, context_tokens.device)

        query = self.decoder_embed(query_tokens)
        context = self.context_embed(context_tokens.detach())
        for block in self.decoder:
            query, _ = block(query, context, query_pos, context_pos)
        return self.output_proj(self.dec_norm(query))


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
        self._lora_attached = False
        # Adapters must exist before FSDP2 wraps the backbone so their
        # parameters are included in the sharded module and optimizer groups.
        if self.lora_enabled:
            from vit_lora import add_lora_to_vit, freeze_non_lora_parameters

            add_lora_to_vit(self.student.backbone, r=self.lora_rank)
            add_lora_to_vit(self.teacher.backbone, r=self.lora_rank)
            freeze_non_lora_parameters(self.student.backbone)
            freeze_non_lora_parameters(self.teacher.backbone)
            self._lora_attached = True

        self.teacher.requires_grad_(False)
        self.model_ema = self.teacher

        self.dino_loss = DINOLoss(self.dino_out_dim)
        self.ibot_patch_loss = iBOTPatchLoss(ibot_out_dim)
        self._distributed_prepared = False
        self.lambda1 = float(cfg.lambda1)
        self.lambda2 = float(cfg.lambda2)
        self.lam_cross = float(cfg.lam_cross)
        self.gamma = float(cfg.gamma)
        self.teacher_temp = float(cfg.teacher.teacher_temp)
        self.mask_ratio_min, self.mask_ratio_max = tuple(cfg.ibot.mask_ratio_min_max)
        self.mask_sample_probability = float(cfg.ibot.mask_sample_probability)
        self.ema_params_lists: Optional[Tuple[list[nn.Parameter], list[nn.Parameter]]] = None

        logger.info(
            "Built slice-pair fine-tuner: embed_dim=%d, dino_prototypes=%d, "
            "ibot_prototypes=%d, gamma=%.3f, lam_cross=%.3f",
            self.embed_dim,
            self.dino_out_dim,
            ibot_out_dim,
            self.gamma,
            self.lam_cross,
        )

    @property
    def cross_view_decoder(self) -> nn.Module:
        return self.student["cvd"]

    def _copy_student_to_teacher(self) -> None:
        student_state = self.student.state_dict()
        teacher_state = {
            name: student_state[name]
            for name in self.teacher.state_dict()
            if name in student_state
        }
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
            if self._lora_attached:
                # The released checkpoint stores fused projections as
                # ``attn.qkv.{weight,bias,bias_mask}``; LoRA wraps that base
                # layer under ``attn.qkv.qkv``.
                for suffix in ("weight", "bias", "bias_mask"):
                    if name.endswith(f".attn.qkv.{suffix}"):
                        name = name.replace(
                            f".attn.qkv.{suffix}", f".attn.qkv.qkv.{suffix}"
                        )
                        break
            if name in expected_backbone_keys:
                backbone_state[name] = value

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
            from vit_lora import init_lora_parameters

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

    def _weighted_ce(self, student_logits: Tensor, teacher_probs: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
        student_temp = float(getattr(self.dino_loss, "student_temp", 0.1))
        log_probs = F.log_softmax(student_logits.float() / student_temp, dim=-1)
        ce = -(teacher_probs.float() * log_probs).sum(dim=-1)
        weights = weights.to(device=ce.device, dtype=ce.dtype)
        return (ce * weights).sum(), weights.sum()

    def _uncertainty_weight(self, teacher_probs: Tensor) -> Tensor:
        entropy = -(teacher_probs.float().clamp_min(1e-8) * teacher_probs.float().clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / math.log(max(teacher_probs.shape[-1], 2))
        return 1.0 + self.gamma * entropy

    def uwsd_loss(
        self,
        teacher1: Dict[str, Tensor],
        teacher2: Dict[str, Tensor],
        student1: Dict[str, Tensor],
        student2: Dict[str, Tensor],
        local_logits1: Optional[Tensor] = None,
        local_logits2: Optional[Tensor] = None,
        teacher_temp: Optional[float] = None,
    ) -> Tensor:
        """UWSD multi-crop DINO loss with slice-aware pair routing."""
        temperature = float(teacher_temp or self.teacher_temp)
        teacher_logits = torch.cat((teacher1["cls_logits"], teacher2["cls_logits"]), dim=0)
        teacher_probs = _sinkhorn_knopp(teacher_logits, temperature)
        batch_size = teacher1["cls_logits"].shape[0]
        teacher_probs1, teacher_probs2 = teacher_probs.split(batch_size, dim=0)

        terms: list[Tuple[Tensor, Tensor, Tensor]] = []
        weight1 = self._uncertainty_weight(teacher_probs1)
        weight2 = self._uncertainty_weight(teacher_probs2)

        if local_logits1 is not None:
            for logits in local_logits1:
                terms.append((logits, teacher_probs1, weight1))
        if local_logits2 is not None:
            for logits in local_logits2:
                terms.append((logits, teacher_probs2, weight2))

        # Cross-slice global-to-global terms are the only global comparison.
        terms.append((student1["cls_logits"], teacher_probs2, self.lam_cross * weight2))
        terms.append((student2["cls_logits"], teacher_probs1, self.lam_cross * weight1))

        numerator: Optional[Tensor] = None
        denominator: Optional[Tensor] = None
        for student_logits, target_probs, weights in terms:
            term_num, term_den = self._weighted_ce(student_logits, target_probs, weights)
            numerator = term_num if numerator is None else numerator + term_num
            denominator = term_den if denominator is None else denominator + term_den
        assert numerator is not None and denominator is not None
        return numerator / denominator.clamp_min(1e-8)

    @staticmethod
    def _masked_values(values: Tensor, masks: Tensor) -> Tensor:
        return values[masks]

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

    def croco_ibot_loss(
        self,
        teacher1: Dict[str, Tensor],
        teacher2: Dict[str, Tensor],
        student1: Dict[str, Tensor],
        student2: Dict[str, Tensor],
        masks1: Tensor,
        masks2: Tensor,
        teacher_temp: Optional[float] = None,
    ) -> Tensor:
        """Cross-view masked-patch completion with same-slice targets."""
        batch_size, num_patches, _ = teacher1["patch_pre_head"].shape
        positions = _patch_positions(batch_size, num_patches, teacher1["patch_pre_head"].device)
        refined1 = self._decode_masked_queries(
            student1["patch_pre_head"],
            teacher2["patch_pre_head"],
            masks1,
            positions,
        )
        refined2 = self._decode_masked_queries(
            student2["patch_pre_head"],
            teacher1["patch_pre_head"],
            masks2,
            positions,
        )

        teacher_selected1 = self._masked_values(teacher1["ibot_logits"], masks1)
        teacher_selected2 = self._masked_values(teacher2["ibot_logits"], masks2)
        teacher_selected = torch.cat((teacher_selected1, teacher_selected2), dim=0)
        teacher_probs = _sinkhorn_knopp(teacher_selected, float(teacher_temp or self.teacher_temp))
        count1 = teacher_selected1.shape[0]
        teacher_probs1, teacher_probs2 = teacher_probs.split((count1, teacher_selected2.shape[0]), dim=0)

        student_logits1 = self.student.ibot_head(refined1)
        student_logits2 = self.student.ibot_head(refined2)
        student_temp = float(getattr(self.ibot_patch_loss, "student_temp", 0.1))

        def patch_ce(student_logits: Tensor, target: Tensor) -> Tensor:
            if student_logits.numel() == 0:
                return student_logits.sum() * 0.0
            return -(target.float() * F.log_softmax(student_logits.float() / student_temp, dim=-1)).sum(-1).mean()

        return 0.5 * (patch_ce(student_logits1, teacher_probs1) + patch_ce(student_logits2, teacher_probs2))

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

        uwsd_loss = self.uwsd_loss(
            teacher1,
            teacher2,
            student1,
            student2,
            local_logits1=local_logits1,
            local_logits2=local_logits2,
            teacher_temp=teacher_temp,
        )
        batch_size = student_global["cls_after_head"].shape[1]
        masks = masks.reshape(2, batch_size, -1)
        croco_ibot_loss = self.croco_ibot_loss(
            teacher1,
            teacher2,
            student1,
            student2,
            masks[0],
            masks[1],
            teacher_temp=teacher_temp,
        )
        total = self.lambda1 * uwsd_loss + self.lambda2 * croco_ibot_loss
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
        """EMA only the student heads and any explicitly named LoRA weights."""
        student_named = dict(self.student.named_parameters())
        teacher_named = dict(self.teacher.named_parameters())
        if self.ema_params_lists is None:
            student_params: list[nn.Parameter] = []
            teacher_params: list[nn.Parameter] = []
            for name, student_param in student_named.items():
                is_head = name.startswith(("dino_head.", "ibot_head."))
                is_lora = any(token in name.lower() for token in ("lora", "w_a_", "w_b_"))
                if (is_head or is_lora) and name in teacher_named:
                    student_params.append(student_param)
                    teacher_params.append(teacher_named[name])
            self.ema_params_lists = (student_params, teacher_params)

        student_params, teacher_params = self.ema_params_lists
        for teacher_param, student_param in zip(teacher_params, student_params):
            teacher_param.mul_(momentum).add_(student_param, alpha=1.0 - momentum)

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
        if bool(optim_cfg.multi_tensor_optim):
            fused_groups = fuse_params_groups(
                params_groups,
                keys=("lr_multiplier", "wd_multiplier", "is_last_layer", "is_lora"),
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
