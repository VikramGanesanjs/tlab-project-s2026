# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import gc
import logging
import math
import os
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_DINOV3_DIR = _SRC_DIR.parent / "opt" / "dinov3"
for _path in (_SRC_DIR, _DINOV3_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.distributed
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job
from dinov3.data import (
    SamplerType,
    make_data_loader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay

from datasets.adni import ADNIPairedSliceDataset, DEFAULT_ROOT as ADNI_DEFAULT_ROOT
from datasets.duke import DukeBreastMRIDataset, PairToDinoGlobalCrops
from datasets.duke.dataset import _DEFAULT_OUT_ROOT as DUKE_DEFAULT_ROOT

try:
    from .model import SSLFineTune
except ImportError:  # Support direct execution: python src/ssl_finetuning/train.py
    from model import SSLFineTune

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

logger = logging.getLogger("dinov3")


def identity_transform(image):
    return image


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


def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
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
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    subgroup_size = distributed.get_subgroup_size()
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * subgroup_size / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * subgroup_size / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * subgroup_size / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * subgroup_size / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    # The custom SSL objective trains both heads from the beginning. Keep the
    # last-layer schedule active during Stage A (heads + CVD only).
    last_layer_lr = lr.copy()
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
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
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if is_last_layer:
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
    dataset_name = cfg.train.dataset.lower()
    data_root = Path(cfg.train.data_root) if cfg.train.data_root else None

    if dataset_name == "adni":
        dataset = ADNIPairedSliceDataset(
            root=data_root or ADNI_DEFAULT_ROOT,
            task=cfg.train.adni_task,
            max_distance=cfg.train.max_distance,
            n_patients=cfg.train.n_patients,
            transform=identity_transform,
            image_size=cfg.crops.global_crops_size,
            seed=cfg.train.seed,
        )
    elif dataset_name == "duke":
        dataset = DukeBreastMRIDataset(
            root=data_root or DUKE_DEFAULT_ROOT,
            scan=cfg.train.duke_scan,
            max_distance=cfg.train.max_distance,
            n_patients=cfg.train.n_patients,
            return_pair=True,
            transform=identity_transform,
            image_size=cfg.crops.global_crops_size,
            seed=cfg.train.seed,
        )
    else:
        raise ValueError(f"Unknown paired dataset={dataset_name!r}; expected 'adni' or 'duke'")

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
    return data_loader


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    del seed
    return build_data_loader_from_cfg(cfg=cfg, model=model, start_iter=start_iter)


def set_backbone_trainable(model, trainable: bool):
    """Toggle only LoRA adapters; the pretrained backbone stays frozen."""
    lora_markers = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
    for name, parameter in model.student.backbone.named_parameters():
        if any(marker in name for marker in lora_markers):
            parameter.requires_grad_(trainable)


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
    ) = build_schedulers(cfg)
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
    global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_subgroup_size()
    freeze_backbone_iters = cfg.optim.freeze_backbone_epochs * OFFICIAL_EPOCH_LENGTH
    backbone_frozen = start_iter < freeze_backbone_iters
    set_backbone_trainable(model, not backbone_frozen)
    logger.info(
        "Backbone LoRA parameters are %s; freeze_backbone_epochs=%d",
        "frozen" if backbone_frozen else "trainable",
        cfg.optim.freeze_backbone_epochs,
    )

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
    consecutive_nan_count = 0
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

        if backbone_frozen and iteration >= freeze_backbone_iters:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            logger.info("Backbone LoRA parameters unfrozen at iteration %d", iteration)

        # Learning rates and other schedules
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

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

        # Reduce total_loss to check for NaNs, reduce metrics for logging
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
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
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)

        # Submit evaluation jobs
        if (
            cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()

        # Checkpointing
        if (iteration + 1) % cfg.checkpointing.period == 0:
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))

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
    cfg = setup_config(args, strict_cfg=False)
    if args.pretrained_weights:
        cfg.student.resume_from_teacher_chkpt = args.pretrained_weights
    elif cfg.student.pretrained_weights and not cfg.student.resume_from_teacher_chkpt:
        cfg.student.resume_from_teacher_chkpt = cfg.student.pretrained_weights
    logger.info(cfg)
    setup_logging(
        output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
        name="nan_logger",
    )
    logger.info("Making slice-pair SSL fine-tuner")
    with torch.device("meta"):
        model = SSLFineTune(cfg)
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
    model.prepare_for_distributed_training()
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
