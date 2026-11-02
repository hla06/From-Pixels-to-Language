# From Pixels to Language

One CNN encoder backs three tasks: image classification, colorization and image captioning, all on
CUB-200-2011 with captions (`Multimodal-Fatima/CUB_train`, `Multimodal-Fatima/CUB_test`).

**Status: Milestone 1 (EDA, classification, encoder) is complete.** Colorization and captioning are
**Milestone 2**; the CLI accepts `--task colorization|caption` but exits with a "Milestone 2" message.

### Milestone 1 results (CUB test split, 5,794 images)

| Model | Top-1 acc. (%) | Macro F1 (%) |
|---|---|---|
| ScratchCNN (from scratch) | 17.31 | 15.47 |
| ResNet-18 (random init) | 37.73 | 36.65 |
| ResNet-18 frozen / fine-tuned | 60.75 / 75.08 | 60.74 / 75.05 |
| ResNet-18 tuned (grid-search best) | 74.34 | 74.09 |
| ResNet-50 frozen / **fine-tuned (final model)** | 63.44 / **80.70** | 63.11 / **80.60** |
| EfficientNet-B0 frozen / fine-tuned | 76.94 / 78.32 | 76.98 / 78.38 |

The final model is `resnet50_finetune`, chosen on validation accuracy (see the report, Section 14).
The full report is `report/Team7_Milestone1_Report.pdf`; failed experiments are in `outputs/NOTES.md`.

### Trained models

All nine checkpoints are on Google Drive:
<https://drive.google.com/drive/folders/10oDTZjIkUaUYYdQJh_dtkMqeObfuFMyq> (the `checkpoints` folders).
They are also stored as W&B artifacts (`<run>-best`), so they can be restored without Drive:

```
python -m src.cli fetch-checkpoints            # needs wandb login; writes outputs/checkpoints/<run>.pt
python -m src.cli fetch-history                # per-epoch curves from W&B into outputs/history/<run>.json
```

## Repository layout

```
src/
  cli.py            command line entry point (eda, train, tune, evaluate, gradcam, predict, figures,
                    fetch-checkpoints, fetch-history)
  config.py         YAML config loading
  data/             CUB loading, caption cleaning, splits, transforms, image statistics
  models/           Encoder interface, registry, ScratchCNN, torchvision encoders, Classifier
  training/         Trainer, tuner, samplers, experiment tracker (W&B), seeding
  evaluation/       metrics, evaluate/predict helpers, benchmark table
  visualization/    EDA plots, Grad-CAM, comparison figures
configs/            one YAML per training run (8 + resnet18_tuned), tune_resnet18.yaml, benchmark.yaml
notebooks/          colab_runner.ipynb (shell cells only; no logic)
outputs/NOTES.md    cleaning steps, findings, failed experiments, M1 coverage and gaps
tests/              pytest suite
```

## Setup

Windows (PowerShell):

```
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
wandb login
python -m pytest tests -v
```

Linux / Colab / Kaggle:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

On Windows keep the venv path short (under about 100 characters): pip fails on long paths unless Windows long-path support is enabled.

`wandb login` is needed for real runs and for `tune` (configs use the W&B tracker). `--smoke` runs and `eda` do not track.
The CUB dataset is downloaded from Hugging Face into the default HF cache on first use.

## Commands

All commands run from the repository root as `python -m src.cli <command> ...`.

EDA: class distribution, image sizes and quality (brightness, contrast, sharpness), sample images with
captions, caption statistics, augmentation examples. It also cleans the captions (lowercase, punctuation
removed, rare typos corrected) into `outputs/data/captions_clean.json` and writes the cleaning steps and
findings block in `outputs/NOTES.md`:

```
python -m src.cli eda --config configs/scratch_cnn.yaml
```

Train, one command per config (the 8 Milestone 1 runs). `--task` defaults to `classification`:

```
python -m src.cli train --task classification --config configs/scratch_cnn.yaml
python -m src.cli train --task classification --config configs/resnet18_random.yaml
python -m src.cli train --task classification --config configs/resnet18_frozen.yaml
python -m src.cli train --task classification --config configs/resnet18_finetune.yaml
python -m src.cli train --task classification --config configs/resnet50_frozen.yaml
python -m src.cli train --task classification --config configs/resnet50_finetune.yaml
python -m src.cli train --task classification --config configs/efficientnet_b0_frozen.yaml
python -m src.cli train --task classification --config configs/efficientnet_b0_finetune.yaml
```

Add `--resume` to `train` or `tune` to skip finished runs and continue an interrupted one (the trainer saves
`<run>.last.pt` after every epoch), for example after a Colab runtime reset.

Smoke test (CPU, 32 images, 1 epoch, no tracker; writes `outputs/smoke/smoke.pt`). It only proves the
pipeline runs; its metrics are meaningless:

```
python -m src.cli train --task classification --config configs/scratch_cnn.yaml --smoke
```

Hyperparameter tuning, Grid Search and Random Search on the same ResNet-18 setup, so they can be
compared (best score, trials, time). Results go to `outputs/tuning/<method>.json`:

```
python -m src.cli tune --method grid   --config configs/tune_resnet18.yaml
python -m src.cli tune --method random --config configs/tune_resnet18.yaml
```

`tune` logs to W&B too: each trial is a run in the group `tune-<method>`, so `wandb login` is needed for it as
well. If a trial crashes, the finished trials are saved to `outputs/tuning/<method>.partial.json`.

After tuning, retrain the best setting at the full budget with `configs/resnet18_tuned.yaml` (it has its own
run name and checkpoint, so `resnet18_finetune` is not overwritten). Copy `lr_backbone` = best lr,
`lr_head` = 10 x best lr, `weight_decay` and `dropout` from `outputs/tuning/<method>.json` into it first:

```
python -m src.cli train --task classification --config configs/resnet18_tuned.yaml
```

Evaluate a checkpoint on the test set (accuracy, precision, recall, F1, confusion matrix, most
confused pairs); `--name` names the result file `outputs/eval/<name>.json`:

```
python -m src.cli evaluate --checkpoint outputs/checkpoints/resnet50_finetune.pt --name resnet50_finetune
```

Grad-CAM on correct and wrong test predictions (grids written to `--out`, default `outputs/figures`):

```
python -m src.cli gradcam --checkpoint outputs/checkpoints/resnet50_finetune.pt --out outputs/figures/gradcam_resnet50_finetune
```

Predict one image (top-k classes; `--gradcam-out` optionally saves a Grad-CAM overlay). `--checkpoint`
defaults to `outputs/checkpoints/resnet50_finetune.pt`, the final model; pass `--checkpoint` to use another
one (for example `resnet18_tuned.pt`; restore them first with `fetch-checkpoints`). A missing image or checkpoint prints an error and exits with code 2:

```
python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint outputs/checkpoints/resnet50_finetune.pt --topk 5 --gradcam-out outputs/figures/bird_cam.png
```

Comparison figures and the benchmark table, built from whatever result files exist in
`outputs/history`, `outputs/eval`, `outputs/tuning` (plus `configs/benchmark.yaml`):

```
python -m src.cli figures
```

## Colab / Kaggle recipe

Open `notebooks/colab_runner.ipynb` on a free T4 runtime. It contains only shell cells: clone, install,
`wandb login`, mount Drive, `eda`, time one epoch, the 8 training runs, both tuning methods, `evaluate`,
`gradcam`, `figures`, and a copy of `outputs/` to Drive. Checkpoints are git-ignored, so copy them to Drive
before the session ends.

## Reproducibility

- `set_seed(cfg["seed"])` (default 42) seeds Python, NumPy and torch (CPU and CUDA), sets
  `PYTHONHASHSEED`, and forces deterministic cuDNN; DataLoader workers are seeded from torch's worker seed.
- `requirements.txt` pins every dependency, including `torch==2.14.1` and `torchvision==0.29.1`.
- If those exact torch versions are not available for your Python/CUDA (common on Colab), keep the preinstalled
  torch and install the rest, then record the torch version you used in `outputs/NOTES.md`:
  `grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt`
- GPU runs are not bit-for-bit reproducible across hardware or torch builds; expect small metric differences.

## Outputs

| Path | Content | Git |
|---|---|---|
| `outputs/NOTES.md` | cleaning steps, findings, failed experiments, M1 coverage and gaps | tracked |
| `outputs/checkpoints/` | trained checkpoints (`*.pt`) | ignored |
| `outputs/smoke/`, `outputs/history/`, `outputs/eval/`, `outputs/tuning/` | smoke checkpoint, per-run curves, test metrics, tuning results | ignored |
| `outputs/figures/`, `outputs/eda/`, `outputs/data/` | figures, EDA plots, derived data | ignored |

Because generated outputs are ignored, copy the figures you need into the report yourself.
W&B holds hyperparameters, metrics, curves and checkpoint references for every real run.

## Design patterns

- **Factory / Registry** (`src/models/registry.py`): encoders register under a name with `@register(name)`;
  callers use `build_encoder(name, ...)`, so a new encoder never touches the training loop.
- **Strategy** (`src/training/samplers.py`, `src/training/tracker.py`): the tuner depends only on a sampler's
  `trials()` (Grid or Random), and training depends only on the tracker interface (W&B or null).
- **Template Method** (`src/training/trainer.py`): `Trainer.fit()` fixes the epoch loop, early stopping and
  checkpointing; the model, optimizer, scheduler and tracker are plugged in.
