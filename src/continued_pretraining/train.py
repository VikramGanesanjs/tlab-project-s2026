# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import copy
import gc
import logging
import math
import os
import sys
from collections import deque
from functools import partial
from pathlib import Path

_CONTINUED_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CONTINUED_DIR.parent
_DINOV3_DIR = _SRC_DIR.parent / "opt" / "dinov3"
for _path in (_SRC_DIR, _DINOV3_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if __package__ in (None, ""):
    __package__ = "continued_pretraining"

import numpy as np
import torch
import torch.distributed
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    register_dont_save_hooks,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.data import (
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from .data import build_dataset_from_cfg
from .ssl_meta_arch import LORA_MARKERS, SSLMetaArch
from utils.fold_cv import dataset_subset_for_patients, make_dataset_patient_folds

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

logger = logging.getLogger("dinov3")


def _adni_patient_split_stratum(dataset, index):
    """Stratify ADNI patients by diagnosis label for the shared splitter."""
    return dataset.get_target(index)


def _build_adni_train_subset(cfg, dataset):
    """Select the class-balanced training subset for one ADNI CV fold."""
    folds = make_dataset_patient_folds(
        dataset,
        n_folds=int(cfg.train.n_folds),
        seed=int(cfg.train.data_seed),
        target_fn=_adni_patient_split_stratum,
    )
    train_patient_ids, val_patient_ids, test_patient_ids = folds.get_split(
        int(cfg.train.fold),
        train_ratio=float(cfg.train.train_ratio),
        train_seed=int(cfg.train.data_seed),
    )
    train_dataset = dataset_subset_for_patients(dataset, train_patient_ids)

    logger.info(
        "ADNI continued pretraining fold %d/%d (seed=%d) uses %d class-balanced "
        "training patients (%d samples, ratio=%.3f); validation/test reserve %d/%d patients",
        int(cfg.train.fold), int(cfg.train.n_folds), int(cfg.train.data_seed),
        len(train_patient_ids), len(train_dataset), float(cfg.train.train_ratio),
        len(val_patient_ids), len(test_patient_ids),
    )
    return train_dataset


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
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def _freeze_iterations(cfg):
    freeze_iterations = int(getattr(cfg, "freeze_iterations", 0))
    if freeze_iterations < 0:
        raise ValueError(f"freeze_iterations must be non-negative, got {freeze_iterations}")
    return freeze_iterations


def _delayed_lora_schedule(cfg, total_iterations):
    freeze_iterations = _freeze_iterations(cfg)
    if freeze_iterations >= total_iterations:
        raise ValueError("freeze_iterations must be shorter than the total training run")
    remaining_iterations = total_iterations - freeze_iterations
    warmup_iterations = int(cfg.optim.warmup_iterations)
    if warmup_iterations < 0:
        raise ValueError("optim.warmup_iterations must be non-negative")
    if warmup_iterations > remaining_iterations:
        raise ValueError("LoRA warmup extends beyond the end of training")
    schedule = CosineScheduler(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=remaining_iterations,
        warmup_iters=warmup_iterations,
        start_warmup_value=0,
        trunc_extra=cfg.optim.schedule_trunc_extra,
    ).schedule
    return np.concatenate((np.zeros(freeze_iterations, dtype=np.float64), schedule))


def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_iterations"],
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    lora_lr_schedule = _delayed_lora_schedule(
        cfg, cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    )
    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        lora_lr_schedule,
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=cfg.schedules.lr.warmup_iterations,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    freeze_iterations = _freeze_iterations(cfg)
    if freeze_iterations >= total_iterations:
        raise ValueError("freeze_iterations must be shorter than the total training run")
    lora_lr = linear_warmup_cosine_decay(
        start=0.0,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=cfg.schedules.lr.warmup_iterations,
        total_iterations=total_iterations - freeze_iterations,
    )
    lora_lr = np.concatenate((np.zeros(freeze_iterations, dtype=np.float64), lora_lr))
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=cfg.schedules.weight_decay.warmup_iterations,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=cfg.schedules.momentum.warmup_iterations,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=cfg.schedules.teacher_temp.warmup_iterations,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr, lora_lr


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


def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    if cfg.multidistillation.enabled:
        assert cfg.multidistillation.global_batch_size % distributed.get_subgroup_size() == 0
        local_batch_size = cfg.multidistillation.global_batch_size // distributed.get_subgroup_size()
        dataloader_batch_size_per_gpu = (
            cfg.multidistillation.global_batch_size + (distributed.get_world_size() - 1)
        ) // distributed.get_world_size()
    else:
        local_batch_size = None  # will default to the standard local batch size matching the data batch size
        dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )
    batch_size = dataloader_batch_size_per_gpu
    num_workers = cfg.train.num_workers
    dataset_path = cfg.train.dataset_path
    if hasattr(cfg.train, "dataset") and cfg.train.dataset:
        dataset = build_dataset_from_cfg(
            cfg,
            transform=model.build_data_augmentation_dino(cfg),
            target_transform=lambda _: (),
        )
    elif dataset_path.split(":", 1)[0] == "ADNI":
        dataset_kwargs = {}
        for token in dataset_path.split(":")[1:]:
            key, value = token.split("=", 1)
            if key not in ("root", "extra", "split"):
                raise ValueError(f"Unsupported ADNI dataset option {key!r}")
            dataset_kwargs[key] = value
        requested_split = str(dataset_kwargs.pop("split", "TRAIN")).strip().upper()
        if requested_split != "TRAIN":
            raise ValueError(
                "Continued pretraining only accepts the ADNI train split; "
                f"received split={requested_split!r}"
            )
        # Legacy ADNI dataset_path support. New configurations should use the
        # generic ``train.dataset`` fields handled above.
        from .adni import ADNI

        dataset = ADNI(transform=model.build_data_augmentation_dino(cfg), target_transform=lambda _: (), **dataset_kwargs)
        dataset = _build_adni_train_subset(cfg, dataset)
    else:
        dataset = make_dataset(
            dataset_str=dataset_path,
            transform=model.build_data_augmentation_dino(cfg),
            target_transform=lambda _: (),
        )

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
        sampler_advance=start_iter * dataloader_batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size] if isinstance(cfg.crops.global_crops_size, int) else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size] if isinstance(cfg.crops.local_crops_size, int) else cfg.crops.local_crops_size
    )
    gram_teacher_crops_sizes = (
        [cfg.crops.gram_teacher_crops_size]
        if cfg.crops.gram_teacher_crops_size is None or isinstance(cfg.crops.gram_teacher_crops_size, int)
        else cfg.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (global_crops_size_i, local_crops_size_i, gram_teacher_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes)
    ):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
        cfg_i.train.seed = cfg.train.seed + increment + 1
        loaders.append(build_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))

    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=seed,
            name="MultiResDL",
        )
    return data_loader


def do_train(cfg, model, resume=False):
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        lora_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    model.init_weights()
    start_iter = 0
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
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    freeze_iterations = _freeze_iterations(cfg)
    checkpoint_period = int(cfg.checkpointing.period)
    if checkpoint_period <= 0:
        raise ValueError(f"checkpointing.period must be positive, got {checkpoint_period}")
    checkpoint_loss_window = int(cfg.checkpointing.get("loss_window_iterations", 100))
    if checkpoint_loss_window <= 0:
        raise ValueError(
            f"checkpointing.loss_window_iterations must be positive, got {checkpoint_loss_window}"
        )
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    # Build data loader
    data_loader = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # Manual garbage collection
    gc.disable()
    gc.collect()

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    consecutive_nan_count = 0
    recent_losses = deque(maxlen=checkpoint_loss_window)
    best_checkpoint_loss = math.inf
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="Training",
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

        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

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

        if it < freeze_iterations:
            for name, parameter in student.backbone.named_parameters():
                if model.is_lora_warmup_backbone_parameter(name, parameter):
                    parameter.grad = None

        # Reduce total_loss to check for NaNs, reduce metrics for logging
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
        total_loss_value = (
            total_loss.full_tensor().item() if isinstance(total_loss, DTensor) else total_loss.detach().item()
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
            if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
            if math.isfinite(total_loss_value):
                recent_losses.append(total_loss_value)
        # Step optimizer
        optimizer.step()
        model.update_ema(mom)

        # [GRAM] Update gram teacher when using gram teacher and frequent updates
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(lora_lr=lora_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)

        # Submit evaluation jobs
        if (
            cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()

        # Checkpointing
        if (iteration + 1) % checkpoint_period == 0:
            if len(recent_losses) < checkpoint_loss_window:
                logger.info(
                    "Skipping checkpoint at iteration %d: only %d/%d recent finite losses are available",
                    iteration,
                    len(recent_losses),
                    checkpoint_loss_window,
                )
            else:
                recent_loss = sum(recent_losses) / len(recent_losses)
                if recent_loss < best_checkpoint_loss:
                    logger.info(
                        "Saving checkpoint at iteration %d: recent %d-iteration loss %.6f "
                        "(previous best %.6f)",
                        iteration,
                        checkpoint_loss_window,
                        recent_loss,
                        best_checkpoint_loss,
                    )
                    torch.cuda.synchronize()
                    save_checkpoint(
                        ckpt_dir / str(iteration),
                        iteration=iteration,
                        model=model,
                        optimizer=optimizer,
                        overwrite=True,
                        process_group=process_subgroup,
                    )
                    best_checkpoint_loss = recent_loss
                    if distributed.is_subgroup_main_process():
                        keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                        if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                            keep_checkpoint_copy(ckpt_dir / str(iteration))
                else:
                    logger.info(
                        "Skipping checkpoint at iteration %d: recent %d-iteration loss %.6f "
                        "is not below previous best %.6f",
                        iteration,
                        checkpoint_loss_window,
                        recent_loss,
                        best_checkpoint_loss,
                    )

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    if args.multi_distillation:
        print("performing multidistillation run")
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
        logger.info("setup_multidistillation done")
        assert cfg.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(cfg)
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    logger.info(f"Making meta arch {meta_arch.__name__}")
    with torch.device("meta"):
        model = meta_arch(cfg)
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
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        model.init_weights()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
