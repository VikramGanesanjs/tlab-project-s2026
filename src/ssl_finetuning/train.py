# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import gc
import json
import logging
import math
import os
import sys
from collections import deque
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_DINOV3_DIR = _SRC_DIR.parent / "opt" / "dinov3"
for _path in (_SRC_DIR, _DINOV3_DIR):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

import torch
import torch.distributed
from torch.distributed._tensor import DTensor
from omegaconf import OmegaConf

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from dinov3.configs import apply_scaling_rules_to_cfg, setup_job
from dinov3.data import (
    SamplerType,
    make_data_loader,
)
from dinov3.logging import MetricLogger, SmoothedValue, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler

from datasets.adni import ADNIPairedSliceDataset, DEFAULT_ROOT as ADNI_DEFAULT_ROOT
from datasets.duke import DukeBreastMRIDataset, PairToDinoGlobalCrops
from datasets.duke.dataset import _DEFAULT_OUT_ROOT as DUKE_DEFAULT_ROOT

from utils.fold_cv import dataset_subset_for_patients, make_dataset_patient_folds

if __package__:
    from .model import SSLFineTune
else:  # Support direct execution: python src/ssl_finetuning/train.py
    from ssl_finetuning.model import SSLFineTune

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

logger = logging.getLogger("dinov3")


def identity_transform(image):
    return image


def load_ssl_config(args):
    """Load only the slice-finetuning config and explicit CLI overrides.

    The upstream ``setup_config`` merges the entire DINOv3 SSL default config,
    which reintroduces legacy GRAM, distillation, KoLeo, and ImageNet fields
    into the saved run configuration. This task has its own complete Vit-B
    config, so merging those unrelated defaults is both unnecessary and
    misleading.
    """
    cfg = OmegaConf.load(args.config_file)
    overrides = list(args.opts or [])
    if args.output_dir is not None:
        overrides.append(f"train.output_dir={os.path.realpath(args.output_dir)}")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))
    logger.info("Loaded slice-finetuning config from %s", args.config_file)
    return cfg


def save_ssl_config(cfg, output_dir):
    """Save the explicit slice-finetuning configuration without DINO defaults."""
    output_path = os.path.join(os.path.abspath(output_dir), "config.yaml")
    OmegaConf.save(config=cfg, f=output_path)
    logger.info("Saved slice-finetuning config: %s", output_path)


class _PairedSliceCollator:
    """Pickleable collator for paired DINO global and local crops."""

    def __init__(self, pair_transform):
        self.pair_transform = pair_transform

    def __call__(self, samples):
        transformed = [self.pair_transform(sample) for sample in samples]
        global_crops = [
            torch.stack([sample["global_crops"][crop_index] for sample in transformed])
            for crop_index in range(2)
        ]
        n_local_crops = len(transformed[0].get("local_crops", ())) if transformed else 0
        local_crops = [
            torch.stack([sample["local_crops"][crop_index] for sample in transformed])
            for crop_index in range(n_local_crops)
        ]
        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
        }


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_dino",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument(
        "--pretrained-weights",
        "--pretrained_weights",
        default="",
        type=str,
        help="DINOv3 teacher checkpoint used to initialize the backbone and heads",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


_CHECKPOINT_LOSS_WINDOW = 100


def _load_checkpoint_loss_state(checkpoint_dir):
    """Restore rolling-loss state saved beside the latest checkpoint."""
    state_path = Path(checkpoint_dir) / "rolling_loss_state.json"
    if not state_path.is_file():
        return deque(maxlen=_CHECKPOINT_LOSS_WINDOW), math.inf
    try:
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        values = [float(value) for value in state.get("loss_window", [])]
        values = values[-_CHECKPOINT_LOSS_WINDOW:]
        lowest = float(state.get("lowest_average_loss", math.inf))
        if not math.isfinite(lowest) and lowest != math.inf:
            raise ValueError("lowest_average_loss is not finite")
        return deque(values, maxlen=_CHECKPOINT_LOSS_WINDOW), lowest
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not restore checkpoint loss state from %s: %s", state_path, exc)
        return deque(maxlen=_CHECKPOINT_LOSS_WINDOW), math.inf


def _save_checkpoint_loss_state(checkpoint_dir, loss_window, lowest_average_loss):
    """Save rolling-loss state beside an accepted checkpoint."""
    state_path = Path(checkpoint_dir) / "rolling_loss_state.json"
    temporary_path = state_path.with_name(f".{state_path.name}.tmp")
    state = {
        "window_size": _CHECKPOINT_LOSS_WINDOW,
        "loss_window": list(loss_window),
        "lowest_average_loss": float(lowest_average_loss),
    }
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(state_path)


def _delayed_lora_schedule(
    *,
    peak_value,
    final_value,
    total_iterations,
    freeze_iterations,
    warmup_iterations,
    trunc_extra,
):
    """Keep LoRA at zero while frozen, then warm it up after unfreezing."""
    if freeze_iterations >= total_iterations:
        raise ValueError(
            "freeze_backbone_epochs must be shorter than the total training run"
        )
    remaining_iterations = total_iterations - freeze_iterations
    if warmup_iterations > remaining_iterations:
        raise ValueError(
            "LoRA warmup extends beyond the end of training: "
            f"warmup_iterations={warmup_iterations}, "
            f"remaining_iterations={remaining_iterations}"
        )
    tail = CosineScheduler(
        base_value=peak_value,
        final_value=final_value,
        total_iters=remaining_iterations,
        warmup_iters=warmup_iterations,
        start_warmup_value=0.0,
        trunc_extra=trunc_extra,
    )
    import numpy as np

    return np.concatenate((np.zeros(freeze_iterations, dtype=np.float64), tail.schedule))


def _epochs_to_iterations(epochs, iterations_per_epoch: int, *, field_name: str) -> int:
    """Convert an epoch-valued schedule to a concrete iteration boundary."""
    epochs = float(epochs)
    if not math.isfinite(epochs) or epochs < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number, got {epochs}")
    # Schedule arrays and range-based training logic require integer lengths.
    # Round to the nearest iteration so fractional epochs retain their intended
    # duration without silently truncating it.
    return int(math.floor(epochs * iterations_per_epoch + 0.5))


def build_schedulers(cfg, iterations_per_epoch):
    total_iterations = cfg.optim["epochs"] * iterations_per_epoch
    freeze_iterations = _epochs_to_iterations(
        cfg.optim["freeze_backbone_epochs"],
        iterations_per_epoch,
        field_name="optim.freeze_backbone_epochs",
    )
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=total_iterations,
        warmup_iters=0,
        start_warmup_value=cfg.optim["lr"],
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=total_iterations,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=total_iterations,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_warmup_iterations = _epochs_to_iterations(
        cfg.teacher["warmup_teacher_temp_epochs"],
        iterations_per_epoch,
        field_name="teacher.warmup_teacher_temp_epochs",
    )
    lora_warmup_iterations = _epochs_to_iterations(
        cfg.optim["warmup_epochs"],
        iterations_per_epoch,
        field_name="optim.warmup_epochs",
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=teacher_warmup_iterations,
        warmup_iters=teacher_warmup_iterations,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    lora_lr_schedule = _delayed_lora_schedule(
        peak_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iterations=total_iterations,
        freeze_iterations=freeze_iterations,
        warmup_iterations=lora_warmup_iterations,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    # The custom SSL objective trains both heads from the beginning. Keep the
    # last-layer schedule active during Stage A (heads + CVD only).
    last_layer_lr_schedule = CosineScheduler(**lr)
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        lora_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, lora_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if param_group.get("is_lora_warmup", False):
            param_group["lr"] = lora_lr * lr_multiplier
        elif is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def do_test(cfg, model, iteration, process_group, do_low_freq=False):
    # dump a sharded checkpoint
    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    if distributed.is_subgroup_main_process():
        eval_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.sharded_eval_checkpoint:
        ckpt_path = eval_dir / "sharded_teacher_checkpoint"
        if distributed.is_subgroup_main_process():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        teacher_backbone = model.model_ema
        save_checkpoint(
            ckpt_dir=ckpt_path, iteration=iteration, model=teacher_backbone, overwrite=True, process_group=process_group
        )
        if not distributed.is_subgroup_main_process():
            return
    else:
        new_state_dict = model.model_ema.state_dict()
        for k, tensor in list(new_state_dict.items()):
            if isinstance(tensor, DTensor):
                new_state_dict[k] = tensor.full_tensor()
        if not distributed.is_subgroup_main_process():
            return
        # save teacher checkpoint
        ckpt_path = eval_dir / "teacher_checkpoint.pth"
        torch.save({"teacher": new_state_dict}, ckpt_path)
        logger.info("Saved eval checkpoint: %s", ckpt_path)


def _duke_patient_split_stratum(dataset, index):
    """Stratify Duke patients by laterality/bilateral phenotype."""
    raw = dataset.get_phenotype_raw(index)
    bilateral = raw.get("bilateral")
    if bilateral is not None and str(bilateral).strip().upper() in {
        "1",
        "1.0",
        "TRUE",
        "YES",
        "BILATERAL",
    }:
        return 2
    location = raw.get("tumor_location")
    if location is None:
        return None
    normalized = str(location).strip().upper()
    if normalized in {"L", "LEFT", "0", "0.0"}:
        return 0
    if normalized in {"R", "RIGHT", "1", "1.0"}:
        return 1
    return None


def _adni_patient_split_stratum(dataset, index):
    """Stratify ADNI patients by the selected diagnosis task."""
    # ADNIPairedSliceDataset.get_target() returns its configured phenotype
    # vector (Group/Sex/Age), not a scalar diagnosis label. Use the raw Group
    # field so this remains valid for every phenotype-column configuration.
    raw = dataset.get_phenotype_raw(index)
    diagnosis = str(raw.get("Group", "")).strip().upper()
    return {"CN": 0, "MCI": 1, "AD": 2}.get(diagnosis)


def _build_train_patient_subset(cfg, dataset, dataset_name):
    stratum_fn = (
        _adni_patient_split_stratum
        if dataset_name == "adni"
        else _duke_patient_split_stratum
    )
    folds = make_dataset_patient_folds(
        dataset,
        n_folds=int(cfg.train.n_folds),
        seed=int(cfg.train.data_seed),
        target_fn=stratum_fn,
    )
    train_patient_ids, val_patient_ids, test_patient_ids = folds.get_split(
        int(cfg.train.fold),
        train_ratio=float(cfg.train.train_ratio),
        train_seed=int(cfg.train.data_seed),
    )
    train_dataset = dataset_subset_for_patients(dataset, train_patient_ids)

    logger.info(
        "Fine-tuning fold %d/%d (seed=%d) uses %d class-balanced training "
        "patients (%d samples, ratio=%.3f); validation/test reserve %d/%d patients",
        int(cfg.train.fold), int(cfg.train.n_folds), int(cfg.train.data_seed),
        len(train_patient_ids), len(train_dataset), float(cfg.train.train_ratio),
        len(val_patient_ids), len(test_patient_ids),
    )
    return train_dataset


def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    dataset_name = cfg.train.dataset.lower()
    data_root = Path(cfg.train.data_root) if cfg.train.data_root else None

    if dataset_name == "adni":
        dataset = ADNIPairedSliceDataset(
            root=data_root or ADNI_DEFAULT_ROOT,
            task=cfg.train.adni_task,
            max_distance=cfg.train.max_distance,
            n_patients=None,
            transform=identity_transform,
            image_size=cfg.crops.global_crops_size,
            seed=cfg.train.seed,
        )
    elif dataset_name == "duke":
        dataset = DukeBreastMRIDataset(
            root=data_root or DUKE_DEFAULT_ROOT,
            scan=cfg.train.duke_scan,
            max_distance=cfg.train.max_distance,
            n_patients=None,
            return_pair=True,
            transform=identity_transform,
            image_size=cfg.crops.global_crops_size,
            seed=cfg.train.seed,
        )
    else:
        raise ValueError(f"Unknown paired dataset={dataset_name!r}; expected 'adni' or 'duke'")

    dataset = _build_train_patient_subset(cfg, dataset, dataset_name)
    pair_transform = PairToDinoGlobalCrops(model.build_data_augmentation_dino(cfg))

    batch_size = cfg.train.batch_size_per_gpu
    num_workers = cfg.train.num_workers

    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if cfg.train.cache_dataset else SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * batch_size,
        drop_last=True,
        collate_fn=_PairedSliceCollator(pair_transform),
    )
    return data_loader, len(dataset)


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    del seed
    return build_data_loader_from_cfg(cfg=cfg, model=model, start_iter=start_iter)


def _period_in_iterations(section, *, epoch_key, iteration_key, iterations_per_epoch):
    """Resolve a schedule period, preferring dataset-relative epoch units."""
    epoch_period = section.get(epoch_key)
    if epoch_period is not None:
        return int(epoch_period) * iterations_per_epoch
    return int(section.get(iteration_key, 0))


def set_backbone_trainable(model, trainable: bool):
    """Keep LoRA visible to FSDP while controlling updates through its LR.

    FSDP2 initializes mixed-precision gradient metadata lazily.  Toggling a
    parameter from ``requires_grad=False`` to ``True`` after the first forward
    leaves an initially frozen FSDP parameter group without dtype metadata and
    can make its bfloat16 gradient incompatible with the float32 sharded
    parameter.  LoRA therefore stays ``requires_grad=True`` for the entire
    run; the scheduler supplies a zero learning rate while it is logically
    frozen.
    """
    del trainable
    lora_markers = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
    for name, parameter in model.student.backbone.named_parameters():
        if any(marker in name for marker in lora_markers):
            parameter.requires_grad_(True)


def discard_frozen_lora_warmup_grads(model) -> None:
    """Prevent delayed LoRA, norm, and patch-embed updates during warm-up."""
    for name, parameter in model.student.backbone.named_parameters():
        if (
            model.is_lora_warmup_backbone_parameter(name, parameter)
            and parameter.grad is not None
        ):
            parameter.grad = None


def do_train(cfg, model, resume=False):
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    start_iter = 0
    loss_window = deque(maxlen=_CHECKPOINT_LOSS_WINDOW)
    lowest_average_loss = math.inf
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
        loss_window, lowest_average_loss = _load_checkpoint_loss_state(last_checkpoint_dir)
        logger.info(
            "Restored checkpoint loss tracking: window=%d/%d lowest_average_loss=%s",
            len(loss_window),
            _CHECKPOINT_LOSS_WINDOW,
            "unset" if lowest_average_loss == math.inf else f"{lowest_average_loss:.6f}",
        )

    # The paired dataset is finite, but its sampler is intentionally infinite.
    # Define one logical epoch as one complete pass of full global batches.
    data_loader, dataset_size = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )
    global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_subgroup_size()
    if dataset_size < global_batch_size:
        raise ValueError(
            "The paired dataset has fewer examples than one global batch: "
            f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
        )
    iterations_per_epoch = dataset_size // global_batch_size
    if iterations_per_epoch <= 0:
        raise ValueError(
            f"Could not form a complete epoch from dataset_size={dataset_size} "
            f"and global_batch_size={global_batch_size}"
        )
    max_iter = cfg.optim.epochs * iterations_per_epoch
    freeze_backbone_iters = _epochs_to_iterations(
        cfg.optim.freeze_backbone_epochs,
        iterations_per_epoch,
        field_name="optim.freeze_backbone_epochs",
    )
    eval_period_iterations = _period_in_iterations(
        cfg.evaluation,
        epoch_key="eval_period_epochs",
        iteration_key="eval_period_iterations",
        iterations_per_epoch=iterations_per_epoch,
    )
    checkpoint_period_iterations = int(cfg.checkpointing.period_iterations)
    if checkpoint_period_iterations < 0:
        raise ValueError(
            "checkpointing.period_iterations must be non-negative, "
            f"got {checkpoint_period_iterations}"
        )
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        lora_lr_schedule,
    ) = build_schedulers(cfg, iterations_per_epoch)

    logger.info(
        "Training data: %d paired examples; per-GPU batch size=%d; "
        "distributed group size=%d; effective batch size=%d",
        dataset_size,
        cfg.train.batch_size_per_gpu,
        distributed.get_subgroup_size(),
        global_batch_size,
    )
    logger.info(
        "Training schedule: %d epochs × %d iterations/epoch = %d total iterations",
        cfg.optim.epochs,
        iterations_per_epoch,
        max_iter,
    )
    logger.info(
        "LR policy: heads/CVD start at %.6g immediately; LoRA stays at 0 for "
        "%d iterations, then warms to the configured peak over %d iterations",
        float(lr_schedule[0]),
        freeze_backbone_iters,
        _epochs_to_iterations(
            cfg.optim.warmup_epochs,
            iterations_per_epoch,
            field_name="optim.warmup_epochs",
        ),
    )
    logger.info(
        "Schedule settings: LoRA warmup=%.3f epochs (%d iterations); "
        "freeze LoRA=%.3f epochs (%d iterations)",
        cfg.optim.warmup_epochs,
        _epochs_to_iterations(
            cfg.optim.warmup_epochs,
            iterations_per_epoch,
            field_name="optim.warmup_epochs",
        ),
        cfg.optim.freeze_backbone_epochs,
        freeze_backbone_iters,
    )
    logger.info(
        "Evaluation period: %d iterations; checkpoint period: %d iterations",
        eval_period_iterations,
        checkpoint_period_iterations,
    )
    backbone_frozen = start_iter < freeze_backbone_iters
    set_backbone_trainable(model, not backbone_frozen)
    logger.info(
        "Backbone LoRA parameters are %s; freeze_backbone_epochs=%.3f "
        "(%d iterations)",
        "frozen" if backbone_frozen else "trainable",
        cfg.optim.freeze_backbone_epochs,
        freeze_backbone_iters,
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # Keep these as current-value meters so the progress line shows the
    # exact epoch position alongside the smoothed training metrics.
    metric_logger.add_meter("epoch", SmoothedValue(window_size=1, fmt="{value:.0f}"))
    metric_logger.add_meter("epoch_iteration", SmoothedValue(window_size=1, fmt="{value:.0f}"))
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Training loop
    student = model.student
    iteration = start_iter
    consecutive_nan_count = 0
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header=(
            f"Training ({cfg.optim.epochs} epochs × "
            f"{iterations_per_epoch} iterations/epoch)"
        ),
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return

        # Garbage collection (trigger manually so it happens on all ranks at the same time)
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        if backbone_frozen and iteration >= freeze_backbone_iters:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            logger.info("Backbone LoRA parameters unfrozen at iteration %d", iteration)

        if iteration % iterations_per_epoch == 0:
            logger.info(
                "Starting epoch %d/%d (iteration %d/%d)",
                iteration // iterations_per_epoch + 1,
                cfg.optim.epochs,
                iteration,
                max_iter,
            )

        # Learning rates and other schedules
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        lora_lr = lora_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, lora_lr)

        # Forward backward
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)

        # Gradient clipping
        if cfg.optim.clip_grad:
            for k, v in student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm.full_tensor().item()
                    if isinstance(grad_norm, torch.distributed.tensor.DTensor)
                    else grad_norm.item()
                )

        # Keep the LoRA parameters registered as trainable for FSDP2's dtype
        # bookkeeping, but do not let the frozen phase update their optimizer
        # state. Their learning rate is already zero in this phase; clearing
        # the gradients also prevents AdamW moments from accumulating before
        # the scheduled unfreeze.
        if backbone_frozen:
            discard_frozen_lora_warmup_grads(model)

        # Reduce total_loss to check for NaNs, reduce metrics for logging
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
        loss_window.append(float(total_loss.item()))
        rolling_loss_100 = (
            sum(loss_window) / len(loss_window)
            if len(loss_window) == _CHECKPOINT_LOSS_WINDOW
            else None
        )
        metrics_values = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
        )
        torch.distributed.all_reduce(
            metrics_values,
            op=torch.distributed.ReduceOp.AVG,
            group=distributed.get_process_subgroup(),
        )
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            if consecutive_nan_count > 2:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
        # Step optimizer
        optimizer.step()
        model.update_ema(mom)

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(lora_lr=lora_lr)
        metric_logger.update(mom=mom)
        metric_logger.update(teacher_temp=teacher_temp)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(
            epoch=iteration // iterations_per_epoch + 1,
            epoch_iteration=iteration % iterations_per_epoch + 1,
        )
        if rolling_loss_100 is not None:
            metric_logger.update(rolling_loss_100=rolling_loss_100)
        metric_logger.update(total_loss=total_loss, **metrics_dict)

        # Submit evaluation jobs
        if (
            eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()

        # Checkpointing
        if checkpoint_period_iterations > 0 and (iteration + 1) % checkpoint_period_iterations == 0:
            if rolling_loss_100 is None:
                logger.info(
                    "Skipping checkpoint at iteration %d: only %d/%d loss values "
                    "are available for the rolling average",
                    iteration,
                    len(loss_window),
                    _CHECKPOINT_LOSS_WINDOW,
                )
            elif rolling_loss_100 < lowest_average_loss:
                torch.cuda.synchronize()
                save_checkpoint(
                    ckpt_dir / str(iteration),
                    iteration=iteration,
                    model=model,
                    optimizer=optimizer,
                    overwrite=True,
                    process_group=process_subgroup,
                )
                lowest_average_loss = rolling_loss_100
                if distributed.is_subgroup_main_process():
                    _save_checkpoint_loss_state(
                        ckpt_dir / str(iteration),
                        loss_window,
                        lowest_average_loss,
                    )
                    keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                    if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                        keep_checkpoint_copy(ckpt_dir / str(iteration))
                logger.info(
                    "Saved improving checkpoint at iteration %d: rolling_loss_100=%.6f",
                    iteration,
                    rolling_loss_100,
                )
            else:
                logger.info(
                    "Skipping checkpoint at iteration %d: rolling_loss_100=%.6f "
                    "is not below best=%.6f",
                    iteration,
                    rolling_loss_100,
                    lowest_average_loss,
                )

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        argv = list(argv)
        if argv and not argv[0].startswith("-"):
            argv = argv[1:]
        args = get_args_parser().parse_args(argv)
    setup_job(output_dir=args.output_dir, seed=args.seed)
    cfg = load_ssl_config(args)
    if args.pretrained_weights:
        cfg.student.resume_from_teacher_chkpt = args.pretrained_weights
    elif cfg.student.pretrained_weights and not cfg.student.resume_from_teacher_chkpt:
        cfg.student.resume_from_teacher_chkpt = cfg.student.pretrained_weights
    save_ssl_config(cfg, args.output_dir)
    # Apply the same batch-size learning-rate scaling used by DINOv3, but do
    # so only after saving the user-facing configuration snapshot.
    apply_scaling_rules_to_cfg(cfg)
    logger.info(cfg)
    setup_logging(
        output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
        name="nan_logger",
    )
    logger.info("Making slice-pair SSL fine-tuner")
    with torch.device("meta"):
        model = SSLFineTune(cfg)
    # DINOv3's distributed setup materializes the meta-device model. Weight
    # initialization must happen after that step; otherwise FSDP2's
    # ``to_empty`` can replace initialized storage with empty values.
    model.prepare_for_distributed_training()
    # Fill all values with `nans` so that we identify
    # non-initialized values
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    model.init_weights()
    model.assert_finite_state("after initialization and checkpoint loading")
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        return do_test(
            cfg,
            model,
            "manual",
            process_group=distributed.get_process_subgroup(),
        )
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
