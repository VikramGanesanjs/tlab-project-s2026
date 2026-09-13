# TLab Summer 2026 Project - Vikram Ganesan

## Adapter Learning of DINO Vision Foundation Models for Medical Images

### Classification performance benchmarking

Pass `--benchmark` to `python -m classification.run` to record a breakdown for
every completed epoch. The console reports DataLoader wait time, host-to-device
transfer, DINO encoder forward passes, the multi-slice transformer/classifier
forward pass, backward/optimizer work, validation wall time, and total epoch
wall time. Results are saved incrementally to `epoch_benchmarks.json` and are
also included in `run_summary.json`.

On CUDA, the compute sections use CUDA events, so their values remain accurate
despite asynchronous kernel launches. Each record additionally includes the
CUDA allocator's end and peak allocated/reserved memory; `process_peak_rss_mb`
is the peak host-memory RSS for the process (and is therefore cumulative).

```bash
python -m classification.run --dataset adni --benchmark ...
```

#### Multi-slice data-loading performance

The multi-slice datasets do materially more CPU work per item than the
single-slice datasets: they decode an entire source volume, normalize/window
it, resample depth and in-plane resolution, optionally run MONAI 3-D
augmentation, and finally expand every slice to ImageNet-normalized RGB.  For
ADNI and CQ500, each multi-slice item intentionally reloads its NIfTI volume
rather than retaining full volumes in a worker cache.  This protects worker
RSS, but makes compressed NIfTI decode and storage latency a recurring cost on
every epoch.

Use `--benchmark` for a short representative run first. If `data_loading_s`
is comparable to or larger than `dino_forward_s`, tune this path before making
model changes. The most useful order of operations is:

1. Sweep `--num-workers` on the target machine (for example `0, 2, 4, 8`),
   keeping batch size and augmentation fixed. More workers can hurt when all
   workers contend for a network filesystem or host RAM; choose the smallest
   setting that removes GPU starvation.
2. Run once with `--no-augment` to separate MONAI's 3-D affine/smoothing cost
   from I/O and preprocessing. If it is the limiting step, use a lighter
   volume transform or move stochastic augmentation to a later, smaller
   representation after validating its effect on accuracy.
3. Keep the dataset on node-local SSD during training when possible. This is
   especially important for `.nii.gz`, whose decompression is repeated by the
   ADNI and CQ500 multi-slice loaders.
4. For repeated experiments, build a versioned cache of the deterministic
   preprocessing keyed by dataset, split, `n_slices`, `image_size`, and
   preprocessing version. A practical cache stores target-resolution,
   single-channel float16 volumes in sharded files; apply random 3-D
   augmentation after reading the cached volume, then perform RGB/ImageNet
   conversion. Do not cache augmented tensors.
5. Do not assume a per-worker full-volume cache will help. The current ADNI
   and CQ500 multi-slice loaders deliberately disable it, and CQ500 has shown
   a major data-processing speedup with that cache removed. With shuffled,
   one-pass volume sampling, cached arrays can create host-memory pressure and
   worse filesystem/page-cache locality without providing useful hits. Treat
   full-volume caching as an experiment to benchmark only on the exact target
   machine and split; merely increasing `num_workers` will not preserve
   decoded volumes across epochs.

Native-depth CQ500 (`--n-slices null`) has a separate quadratic-cost hazard:
the slice transformer processes a padded sequence up to the deepest volume in
each batch. Bucket scans by depth (or use a fixed `--n-slices`) to reduce both
CPU padding/collation and transformer work. When diagnosing throughput, also
account for validation: it re-executes the full input pipeline every epoch and
is reported separately as `validation_s`.

### Patch-feature PCA visualization

`src/utils/pca_dino_backbones.py` constructs the selected repository single-slice
dataset, samples one slice, and compares DINOv3, BrainDINO, and a
multi-slice-classification checkpoint. The custom checkpoint may be a distributed
checkpoint directory, a plain merged teacher `.pth`, or a
`--hub-compatible` merged teacher `.pth`. The PCA uses `whiten=True`, matching
the DINOv3 reference notebook. Patch features remain in CPU memory while the
next model is loaded.

```bash
python -m utils.pca_dino_backbones \
  --dataset amos \
  --dinov3-checkpoint /path/to/dinov3_vitb16.pth \
  --braindino-checkpoint /path/to/brain_dino_weights.pth \
  --custom-checkpoint /path/to/distributed/checkpoint-or-merged-backbone.pth \
  --dinov3-repo /path/to/dinov3 \
  --image-size 512 \
  --output pca_slice.png
```

`--dataset` selects the standard data root and matching single-slice dataset.
Available values are `adni`, `duke`, `cq500`, `breastdm`, `amos`, and
`brats_men`; use `--data-root` only for a nonstandard data location. BraTSMen's
four MRI modalities are deterministically projected to DINO RGB as T1c, T1n,
and mean(T2f, T2w).

### Classification visualization

`src/utils/classification_visualization.py` samples labeled ADNI slices by default,
extracts CLS tokens from a regular or distributed checkpoint, embeds them with
2-D cosine UMAP, and plots one diagnosis-colored dot per slice.
Sampling prefers one slice per patient; additional slices are only reused when
the requested count exceeds the number of available patients.

```bash
python -m utils.classification_visualization \
  --checkpoint /path/to/checkpoint \
  --dinov3-repo /path/to/dinov3 \
  --data-root /path/to/data/ADNI \
  --n-slices 100 \
  --output classification_umap.png
```

Use `--dataset duke` for Duke data. The default `--encoder auto` detects
BrainDINO and DINOv3 `.pth` files; use `--encoder custom` for a regular custom
checkpoint or `--encoder dinov3` / `--encoder braindino` to force a format.
Use `--seed` for reproducible sampling.

### Four-backbone feature comparison

`src/utils/features_comparison.py` samples the same patient-diverse ADNI slices for
DINOv3, BrainDINO, extended pretraining, and 3-D-aware fine tuning. The
DINOv3 and BrainDINO paths are fixed to the repository defaults; supply only
the two adapted checkpoints. Both adapted-checkpoint arguments accept a
distributed checkpoint directory or a merged teacher `.pth` export.

```bash
python -m utils.features_comparison \
  --data-root /path/to/data/ADNI \
  --extended-pretraining-checkpoint /path/to/extended-pretraining-checkpoint \
  --three-d-aware-finetuning-checkpoint /path/to/3d-aware-finetuning-checkpoint \
  --n-images 5 \
  --image-size 512 \
  --output features_comparison.png
```

To compare a parent folder of distributed checkpoints, use the `evolution`
subcommand. It defaults to five sampled images and lays out images as rows and
checkpoints as columns, with the original image in the leftmost column:

```bash
python -m utils.pca_dino_backbones evolution \
  --checkpoint-parent /path/to/checkpoint_parent \
  --dataset cq500 \
  --n-images 5 \
  --checkpoint-stride 3 \
  --image-size 512 \
  --output pca_checkpoint_evolution.png
```

`--checkpoint-stride N` keeps every Nth checkpoint in discovery order,
starting with the first; `--checkpoint-stride 3` therefore retains entries 0,
3, 6, and so on. The final partial stride is valid.

### Token metrics

`src/utils/token_metrics.py metrics` computes CLS-token effective rank, patch-token
effective rank, mean off-diagonal patch-Gram similarity, and spatial
specificity for the selected distributed checkpoints. It writes a tidy CSV and
a five-panel PNG; spatial specificity plots patch-token similarity against
Euclidean patch-grid distance for each checkpoint. Background patches are
excluded by default; pass `--no-mask-background` to retain them.
The fifth panel plots the per-checkpoint Pearson correlation between patch-pair
distance and similarity against training iteration.

```bash
python -m utils.token_metrics metrics \
  --checkpoint-parent /path/to/checkpoint_parent \
  --data-root /path/to/data/ADNI \
  --checkpoint-stride 3 \
  --n-images 5 \
  --image-size 224 \
  --output token_metrics.csv \
  --plot token_metrics.png
```

Use `comparison` instead to calculate only patch-Gram distance against
`--reference-encoder dinov3` (default) or `--reference-encoder braindino`.

### Training curves

`src/utils/plot_training_logs.py` parses DINOv3 training records and plots losses,
learning rates, gradient norms, and timing against global iteration. Each loss
and gradient norm gets its own subplot, while loss fields are discovered from
the log so custom loss terms are included automatically.

The script accepts the newline-delimited JSON metrics produced by training:

```bash
MPLBACKEND=Agg python -m utils.plot_training_logs \
  runs/continued_pretraining/adni-fixed/training_metrics.json \
  --output runs/training_curves.png --smooth 5 --no-show
```

Use `--value current` for the per-iteration values (the default). The JSONL
metrics contain per-iteration values; `--value average` remains available for
the legacy text logs that include running averages in parentheses.
