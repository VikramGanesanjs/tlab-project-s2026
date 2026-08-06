# TLab Summer 2026 Project - Vikram Ganesan

## Adapter Learning of DINO Vision Foundation Models for Medical Images

### Patch-feature PCA visualization

`src/pca_dino_backbones.py` constructs the repository ADNI dataset, samples one
slice from the middle 50% of a volume, and compares DINOv3, BrainDINO, and a
dino_mst-style checkpoint. The PCA uses `whiten=True`, matching the DINOv3
reference notebook. Patch features remain in CPU memory while the next model
is loaded.

```bash
python src/pca_dino_backbones.py \
  --data-root /path/to/data/ADNI \
  --dinov3-checkpoint /path/to/dinov3_vitb16.pth \
  --braindino-checkpoint /path/to/brain_dino_weights.pth \
  --custom-checkpoint /path/to/distributed/checkpoint \
  --dinov3-repo /path/to/dinov3 \
  --image-size 512 \
  --output pca_adni_slice.png
```

### Classification visualization

`src/classification_visualization.py` samples labeled ADNI slices by default,
extracts CLS tokens from a regular or distributed checkpoint, embeds them with
2-D cosine UMAP, and plots one diagnosis-colored dot per slice.
Sampling prefers one slice per patient; additional slices are only reused when
the requested count exceeds the number of available patients.

```bash
python src/classification_visualization.py \
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

To compare a parent folder of distributed checkpoints, use the `evolution`
subcommand. It defaults to five sampled images and lays out images as rows and
checkpoints as columns, with the original image in the leftmost column:

```bash
python src/pca_dino_backbones.py evolution \
  --checkpoint-parent /path/to/checkpoint_parent \
  --data-root /common/ganesanv/tlab/data/ADNI \
  --n-images 5 \
  --image-size 512 \
  --output pca_checkpoint_evolution.png
```

### Training curves

`src/plot_training_logs.py` parses DINOv3 training records and plots losses,
learning rates, gradient norms, and timing against global iteration. Each loss
and gradient norm gets its own subplot, while loss fields are discovered from
the log so custom loss terms are included automatically.

The script accepts the newline-delimited JSON metrics produced by training:

```bash
MPLBACKEND=Agg python src/plot_training_logs.py \
  runs/continued_pretraining/adni-fixed/training_metrics.json \
  --output runs/training_curves.png --smooth 5 --no-show
```

Use `--value current` for the per-iteration values (the default). The JSONL
metrics contain per-iteration values; `--value average` remains available for
the legacy text logs that include running averages in parentheses.
