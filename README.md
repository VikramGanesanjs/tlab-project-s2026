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
  --output pca_adni_slice.png
```

### Training curves

`src/plot_training_logs.py` parses DINOv3 training records and plots losses,
learning rates, gradient norms, and timing against global iteration. Each loss
and gradient norm gets its own subplot, while loss fields are discovered from
the log so custom loss terms are included automatically.

```bash
MPLBACKEND=Agg python src/plot_training_logs.py \
  runs/continued_pretraining/adni/5353170_0_log.out \
  runs/ssl_finetuning/adni_vitb16/5334042/5334042_0_log.out \
  --output runs/training_curves.png --value average --smooth 5 --no-show
```

Use `--value current` for the per-iteration values (the default), or
`--value average` for the running averages printed in parentheses.
