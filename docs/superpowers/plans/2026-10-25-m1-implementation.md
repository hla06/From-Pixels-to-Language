# Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the M1 pipeline for CUB-200-2011: EDA, a swappable-encoder classifier trained from scratch and by transfer learning, Grid/Random tuning, Grad-CAM, evaluation, and a CLI, tracked in Weights & Biases.

**Architecture:** Encoders share one `Encoder` contract and are built by name from a registry. A `Classifier` (encoder + shared head) is trained by a `Trainer` that never imports a concrete encoder. Tracker and search samplers are Strategy objects. All logic is pure `.py` behind `python -m src.cli`.

**Tech Stack:** Python 3.13, PyTorch + torchvision, Hugging Face `datasets`, Weights & Biases, scikit-learn, matplotlib, pandas, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-m1-design.md` (read it first; this plan refers to its section numbers).

## How to work through this plan

There is no day-by-day schedule. Tasks are one ordered batch; do them in order. Tasks 1 to 8 are the minimum needed to start GPU runs, so start Task 12 (runs) as soon as Task 8 passes, and do Tasks 9 to 11 while the runs execute. Each task ends in a commit. Commit under the member's own git identity. Do not push; the user pushes. Tests never download data or weights: they use tiny synthetic data and `pretrained=False`.

## Global Constraints

- Python and PyTorch; final code is pure `.py` run through the CLI; notebooks are experiments only.
- Seed everything; a run reproduces from `README.md` and `requirements.txt` (pinned versions) alone.
- Data: `Multimodal-Fatima/CUB_train` (5,994 images) and `Multimodal-Fatima/CUB_test` (5,794 images), 200 classes.
- Validation: stratified 80/20 split of train, seed 42, about 4,795 train and 1,199 validation. Test set is used only for final numbers.
- Input size 224; train transform: resize about 256, random resized crop 224, horizontal flip, small rotation, mild colour jitter; no vertical flip. Val/test: resize and center crop. ImageNet mean/std normalisation for all models.
- Head: `Dropout -> Linear(feature_dim, 200)`, shared by all encoders.
- Loss: cross-entropy with label smoothing 0.1, fixed in all runs. Optimizer AdamW, cosine decay with warmup, lower backbone lr than head lr when fine-tuning. AMP on CUDA only. Early stopping and best checkpoint by validation accuracy.
- Tracker: Weights & Biases behind a `Tracker` interface; every run logs hyperparameters, per-epoch metrics, curves, and a checkpoint reference.
- Tuning: ResNet-18 pretrained fine-tuned; lr in {1e-4, 3e-4, 1e-3} x wd in {1e-4, 1e-2} x dropout in {0.2, 0.5} = 12 grid trials; Random = 12 trials (lr, wd log-uniform over the same ranges, dropout uniform in [0.2, 0.5]); about 8 epochs per trial; same seed per trial; score = validation accuracy; best setting retrained once at full budget.
- The 8 runs: ScratchCNN; ResNet-18 random weights; ResNet-18, ResNet-50, EfficientNet-B0 each frozen and fine-tuned.
- Metrics: accuracy, top-5 accuracy, macro precision, macro recall, macro F1, confusion matrix.
- Published benchmark numbers are looked up with a source URL and never quoted from memory.
- Every cleaning step and every failed experiment is logged in `outputs/NOTES.md`.
- Fit a free Colab/Kaggle T4. CLI shape is final from the start: `--task colorization|caption` exist and report "Milestone 2".

## Review Focus

Failure modes the spec implies; each has a test in the owning task:
1. Grayscale, RGBA or palette images (CUB contains grayscale) must become 3-channel tensors, in training and in `predict` (Tasks 3, 10).
2. `description` with trailing newline, blank lines, or a count other than 10 captions must not crash or silently pass (Task 4).
3. A frozen backbone's BatchNorm must stay in eval mode after `model.train()` (Task 5).
4. A checkpoint saved on GPU must load on a CPU-only machine, and `predict` on a missing image path must fail with a clear message (Tasks 7, 10).
5. A class with a single image must not break the stratified split, and an unknown `--task` must exit non-zero with a clear message (Tasks 3, 8).

## File Structure

```
.gitignore  requirements.txt  README.md
configs/    (9 yaml files, Task 8 and 9)
src/__init__.py  src/config.py  src/cli.py
src/data/        splits.py transforms.py cub_dataset.py captions.py image_stats.py
src/models/      base.py registry.py scratch_cnn.py torchvision_encoders.py classifier.py
src/training/    seed.py tracker.py trainer.py samplers.py tuner.py
src/evaluation/  metrics.py evaluate.py benchmark.py
src/visualization/ eda_plots.py gradcam.py comparison.py
tests/           one test file per module
notebooks/colab_runner.ipynb   outputs/NOTES.md
```

---

### Task 1: Scaffold, config loader, seeding

**Files:**
- Create: `.gitignore`, `requirements.txt`, `src/__init__.py` plus `__init__.py` in each `src/` subpackage and `tests/`, `src/config.py`, `src/training/seed.py`, `tests/test_config.py`, `tests/test_seed.py`, `outputs/NOTES.md`
- Modify: `README.md` (setup section only for now)

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict` (YAML to dict); `set_seed(seed: int) -> None`; `seed_worker(worker_id: int) -> None` for DataLoader.

- [ ] **Step 1:** Create a venv (`py -3 -m venv .venv`), activate it, install `torch torchvision datasets wandb pyyaml scikit-learn matplotlib pandas pillow numpy imagehash pytest`. Record exact versions with `pip freeze` filtered to these packages into `requirements.txt` (pinned `==`).
- [ ] **Step 2:** Write `.gitignore` covering `.venv/`, `__pycache__/`, `wandb/`, `.env`, `outputs/checkpoints/`, `data/`, `*.pt`, `*.pth`, `.ipynb_checkpoints/`. Create `outputs/NOTES.md` with headings "Cleaning steps" and "Failed experiments".
- [ ] **Step 3: Write failing tests.** `test_config.py::test_load_config_roundtrip` writes a YAML with nested keys to `tmp_path` and asserts `load_config` returns the same dict. `test_seed.py::test_set_seed_reproducible` calls `set_seed(1)` then `torch.rand(3)`, again, and asserts equal tensors; also asserts `random.random()` and `numpy.random.rand()` repeat.
- [ ] **Step 4:** Run `python -m pytest tests -v`; expected FAIL (modules missing).
- [ ] **Step 5:** Implement `load_config` and `set_seed` (seeds `random`, `numpy`, `torch`, `torch.cuda`, sets `cudnn.deterministic=True`, `cudnn.benchmark=False`) and `seed_worker`.
- [ ] **Step 6:** Run tests; expected PASS.
- [ ] **Step 7:** Commit: `git add .gitignore requirements.txt src tests outputs/NOTES.md README.md && git commit -m "chore: scaffold project, config loader, seeding"`.

### Task 2: Tracker strategy

**Files:**
- Create: `src/training/tracker.py`, `tests/test_tracker.py`

**Interfaces:**
- Produces: `class Tracker(ABC)` with `log_params(params: dict) -> None`, `log_metrics(metrics: dict, step: int) -> None`, `log_artifact(path: str, name: str) -> None`, `finish() -> None`; `NullTracker(Tracker)` that stores calls in `self.calls: list[tuple]`; `WandbTracker(project: str, run_name: str, group: str | None = None, config: dict | None = None)`; `build_tracker(cfg: dict) -> Tracker` reading `cfg["tracker"]["name"]` in `{"wandb", "null"}`.

- [ ] **Step 1: Write failing tests.** `test_null_tracker_records` asserts a `log_metrics({"loss": 1.0}, step=2)` call appears in `calls`. `test_build_tracker_null` asserts `build_tracker({"tracker": {"name": "null"}})` is a `NullTracker`. `test_build_tracker_unknown` asserts `ValueError` for name `"foo"`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. `WandbTracker` imports `wandb` lazily inside `__init__` so tests never need a login. `log_artifact` creates a `wandb.Artifact` of type `"model"` and logs a reference (path) when the file exceeds 100 MB, otherwise the file.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: tracker interface with W&B and null implementations`.

### Task 3: Splits, transforms, dataset

**Files:**
- Create: `src/data/splits.py`, `src/data/transforms.py`, `src/data/cub_dataset.py`, `tests/test_splits.py`, `tests/test_transforms.py`, `tests/test_cub_dataset.py`

**Interfaces:**
- Produces: `stratified_split(labels: Sequence[int], val_fraction: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]` (train_idx, val_idx); `build_transforms(train: bool, image_size: int = 224, aug: dict | None = None) -> Callable[[PIL.Image], Tensor]`; `class CubDataset(Dataset)` with `__init__(hf_split, indices: Sequence[int], transform)` returning `(Tensor[3,H,W], int)`; `load_cub(cfg: dict) -> tuple[Dataset, Dataset, Dataset]` (train, val, test) plus class names; `build_loaders(cfg: dict) -> dict[str, DataLoader]` with keys `train`, `val`, `test`.

- [ ] **Step 1: Write failing tests.** Splits: `test_split_disjoint_and_complete` (union equals all indices, no overlap); `test_split_every_class_in_both` on 10 classes x 30 images (val size 6 per class); `test_split_deterministic_by_seed`; `test_single_image_class_goes_to_train` (a class with 1 image appears only in train, no exception). Transforms: `test_gray_and_rgba_to_three_channels` feeds PIL `"L"` and `"RGBA"` images and asserts output shape `(3, 224, 224)`; `test_eval_transform_deterministic` asserts identical output on two calls; `test_train_transform_stochastic` asserts two calls differ. Dataset: `test_cub_dataset_len_and_item` using a fake HF split (list of dicts with `image`, `label`) asserts length equals `len(indices)` and item types.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3: Implement** `stratified_split` with `sklearn.model_selection.train_test_split(stratify=...)` per the single-image rule above. `build_transforms` begins with `.convert("RGB")`, then spec-section-3 augmentations (colour jitter brightness/contrast/saturation 0.2, hue 0.02), ImageNet normalisation. `load_cub` calls `datasets.load_dataset` for both repos and builds the split from the train labels.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: stratified split, transforms and CUB dataset`.

### Task 4: Caption cleaning, image stats, EDA command

**Files:**
- Create: `src/data/captions.py`, `src/data/image_stats.py`, `src/visualization/eda_plots.py`, `tests/test_captions.py`, `tests/test_image_stats.py`
- Modify: `src/cli.py` (the `eda` subcommand wiring lives in Task 8; here expose `run_eda(cfg: dict) -> None` in `src/visualization/eda_plots.py`)

**Interfaces:**
- Produces: `split_captions(description: str) -> list[str]`; `clean_caption(text: str) -> str`; `caption_stats(captions: list[list[str]]) -> dict` with keys `n_images, n_not_ten, mean_len, vocab_size, rare_words, duplicates`; `write_clean_captions(records, path) -> None` (JSON `{file_name: [captions]}`); `image_stats(split) -> pandas.DataFrame` with columns `width, height, mode, phash`; `find_duplicates(df) -> list[tuple[int, int]]`; plot functions `plot_class_distribution`, `plot_image_sizes`, `plot_caption_lengths`, `plot_augmentations`, each `(data, out_path: Path) -> None`.

- [ ] **Step 1: Write failing tests.** `test_split_captions_drops_blank_and_trailing` (`"a\n\nb\n"` -> `["a", "b"]`); `test_clean_caption_lowercases_and_collapses_spaces` (`"  A  Red   BIRD. "` -> `"a red bird."`); `test_stats_flags_non_ten` (an image with 9 captions raises `n_not_ten == 1`); `test_find_duplicates_same_hash` and `test_image_stats_reports_mode` on tiny PIL images including an `"L"` image.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. `clean_caption`: lowercase, strip, collapse whitespace; do not remove punctuation or fix typos automatically. `run_eda` computes stats, writes figures to `outputs/eda/`, writes `outputs/data/captions_clean.json`, and appends each applied cleaning step and any finding (typos, duplicates, grayscale count) to `outputs/NOTES.md` under "Cleaning steps". Any further typo or noise fix is added only after reading the stats output, then recorded in NOTES.md.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: caption cleaning, image stats and EDA plots`.

### Task 5: Encoders, registry, classifier

**Files:**
- Create: `src/models/base.py`, `src/models/registry.py`, `src/models/scratch_cnn.py`, `src/models/torchvision_encoders.py`, `src/models/classifier.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `class Encoder(nn.Module)` with attribute `feature_dim: int`, `forward(x) -> Tensor[B, feature_dim]`, `gradcam_layer() -> nn.Module`; `register(name: str)` decorator; `build_encoder(name: str, pretrained: bool = True, freeze: bool = False) -> Encoder` (raises `KeyError` listing known names); registered names `scratch_cnn` (512), `resnet18` (512), `resnet50` (2048), `efficientnet_b0` (1280); `class Classifier(nn.Module)` with `__init__(encoder: Encoder, num_classes: int = 200, dropout: float = 0.5)`, `forward(x) -> logits[B, num_classes]`, and `train(mode)` override.

- [ ] **Step 1: Write failing tests** (all with `pretrained=False`): `test_output_shape_and_feature_dim` parametrised over the four names with input `(2, 3, 224, 224)`; `test_gradcam_layer_is_submodule`; `test_unknown_name_lists_known` (`KeyError` message contains `resnet18`); `test_frozen_encoder_no_grad_and_bn_eval` (after `freeze=True`, no encoder parameter requires grad; after `model.train()` every encoder BatchNorm module has `training == False` while the head's dropout has `training == True`); `test_head_shape` (logits `(2, 200)`).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3: Implement.** ScratchCNN: 5 blocks Conv3x3-BatchNorm-ReLU-MaxPool with widths 32, 64, 128, 256, 512, then `AdaptiveAvgPool2d(1)` and flatten; `gradcam_layer` is the last conv. Torchvision encoders: load weights via the `weights=` enum when `pretrained`, drop the classifier, keep global pooling; `gradcam_layer` is `layer4` (ResNets) or `features[-1]` (EfficientNet). `Classifier.train()` calls `super().train(mode)` then, if the encoder is frozen, sets encoder BatchNorm modules to eval.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: encoder contract, registry, scratch and torchvision encoders, classifier`.

### Task 6: Metrics

**Files:**
- Create: `src/evaluation/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray | None = None, num_classes: int = 200) -> dict` with keys `accuracy, top5_accuracy, precision_macro, recall_macro, f1_macro` (`top5_accuracy` only when logits given); `confusion(y_true, y_pred, num_classes: int = 200) -> np.ndarray`; `most_confused_pairs(cm: np.ndarray, k: int = 10) -> list[tuple[int, int, int]]` as (true, pred, count).

- [ ] **Step 1: Write failing tests** with a 3-class example: `test_accuracy_known` (4 of 5 correct -> 0.8); `test_macro_f1_handles_absent_class` (a class never in `y_true` gives no exception and `zero_division=0`); `test_top5_with_logits`; `test_confusion_diagonal_sum_equals_correct`; `test_most_confused_excludes_diagonal`.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement with scikit-learn (`zero_division=0`). **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: classification metrics and confusion helpers`.

### Task 7: Trainer, optimizer, checkpoints

**Files:**
- Create: `src/training/trainer.py`, `tests/test_trainer.py`

**Interfaces:**
- Consumes: `Classifier` (Task 5), `Tracker` (Task 2), `compute_metrics` (Task 6), `set_seed` (Task 1).
- Produces: `build_optimizer(model: Classifier, cfg: dict) -> torch.optim.Optimizer` (AdamW; param groups with `cfg["train"]["lr_backbone"]` and `cfg["train"]["lr_head"]`, `weight_decay`); `build_scheduler(optimizer, cfg: dict, steps_per_epoch: int)` (linear warmup then cosine); `class Trainer` with `__init__(model, loaders: dict[str, DataLoader], optimizer, scheduler, tracker: Tracker, device: str, cfg: dict)` and `fit() -> dict` returning `{"best_val_acc": float, "history": list[dict], "checkpoint": str}`; `load_checkpoint(path: str, device: str = "cpu") -> Classifier`. Checkpoint dict keys: `model_state`, `config`, `class_names`, `val_acc`.

- [ ] **Step 1: Write failing tests** on CPU with a tiny `scratch_cnn` and random-tensor DataLoaders (3 classes, 24 images at 64x64, `num_classes=3`): `test_fit_logs_each_epoch` (a `NullTracker` has one `log_metrics` call per epoch with keys `train_loss, val_loss, train_acc, val_acc, lr`); `test_best_checkpoint_saved_and_loadable` (file exists; `load_checkpoint` returns a model producing the same logits as the trained one); `test_checkpoint_loads_with_map_location_cpu` (save, reload with `device="cpu"`); `test_same_seed_same_history` (two runs after `set_seed(0)` produce equal `history`); `test_early_stopping_stops` (`patience=1` with constant val acc stops before `epochs`); `test_label_smoothing_applied` (the loss function's `label_smoothing == 0.1`).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3: Implement** `Trainer.fit` as a template method with overridable `train_epoch`, `validate`, `save_checkpoint`. AMP via `torch.autocast`/`GradScaler` enabled only when `device == "cuda"`. Keep the best checkpoint by `val_acc`, upload it with `tracker.log_artifact`, and log params once at start.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: trainer with AMP, early stopping and checkpointing`.

### Task 8: CLI (train, router) and run configs

**Files:**
- Create: `src/cli.py`, `configs/scratch_cnn.yaml`, `configs/resnet18_random.yaml`, `configs/resnet18_frozen.yaml`, `configs/resnet18_finetune.yaml`, `configs/resnet50_frozen.yaml`, `configs/resnet50_finetune.yaml`, `configs/efficientnet_b0_frozen.yaml`, `configs/efficientnet_b0_finetune.yaml`, `tests/test_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; `run_train(cfg: dict, loaders: dict | None = None) -> dict` (builds seed, tracker, loaders unless injected, classifier, optimizer, scheduler, `Trainer`; returns the `fit()` result). Subcommands: `eda`, `train`, `tune`, `predict`, `evaluate`, `gradcam`; `--task` choices `classification|colorization|caption`; `train` accepts `--smoke` (32-image subset, 1 epoch, null tracker, CPU). Config keys: `seed, encoder{name, pretrained, freeze}, data{...}, train{epochs, patience, lr_backbone, lr_head, weight_decay, dropout, batch_size, label_smoothing}, tracker{name, project}`.

- [ ] **Step 1: Write failing tests.** `test_all_configs_resolve_encoder` loads every `configs/*.yaml` that has an `encoder` key and asserts `build_encoder(name, pretrained=False)` works; `test_colorization_exits_nonzero_with_message` (`main(["train","--task","colorization","--config","x"])` returns 2 and prints "Milestone 2"); `test_unknown_task_exits_nonzero`; `test_run_train_with_injected_loaders` (tiny loaders, null tracker, 1 epoch, returns `best_val_acc`).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement with `argparse` subparsers. Write the 8 configs: defaults `epochs: 30`, `patience: 7`, `label_smoothing: 0.1`, `batch_size: 64`; scratch and random use `lr_backbone = lr_head = 1e-3`; frozen runs use `lr_head: 1e-3`; fine-tuned runs use `lr_backbone: 1e-4`, `lr_head: 1e-3`. These epoch/lr defaults are provisional and are adjusted in Task 12 after measuring real epoch time on the T4 (spec section 11).
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: CLI train command and eight run configs`.

### Task 9: Tuning (samplers, tuner, comparison)

**Files:**
- Create: `src/training/samplers.py`, `src/training/tuner.py`, `configs/tune_resnet18.yaml`, `tests/test_tuning.py`
- Modify: `src/cli.py` (`tune` subcommand)

**Interfaces:**
- Consumes: `run_train` (Task 8).
- Produces: `class Sampler(ABC)` with `trials() -> Iterator[dict]`; `GridSampler(space: dict[str, list])`; `RandomSampler(space: dict[str, dict], n_trials: int, seed: int)` where each space entry is `{"type": "log_uniform"|"uniform", "low": float, "high": float}`; `@dataclass TuneResult(method: str, best_score: float, best_params: dict, n_trials: int, seconds: float, scores: list[float])`; `Tuner(base_cfg: dict, sampler: Sampler, method: str, train_fn: Callable[[dict], float])` with `run() -> TuneResult`; `plot_tuning_comparison(results: dict[str, TuneResult], out: Path) -> None` in `src/visualization/comparison.py` (Task 11 owns that file; this task adds only this function).

- [ ] **Step 1: Write failing tests.** `test_grid_has_twelve_unique_trials` using the spec grid; `test_random_twelve_trials_within_ranges` (lr between 1e-4 and 1e-3, wd between 1e-4 and 1e-2, dropout between 0.2 and 0.5); `test_random_reproducible_by_seed`; `test_log_uniform_not_linear` (median of 2000 samples of lr is below the arithmetic midpoint); `test_tuner_picks_max_score` (fake `train_fn` returning a known score per params; asserts `best_params`, `n_trials`, and `len(scores)`).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. `Tuner` copies `base_cfg`, applies trial params into `train.lr_backbone`, `train.weight_decay`, `train.dropout` (lr applies to the backbone and `lr_head = 10 x lr_backbone`), sets `train.epochs: 8`, names each W&B run `"{method}-trial-{i}"` and groups by method. The CLI `tune --method grid|random` builds the sampler from `configs/tune_resnet18.yaml`, runs the tuner with `train_fn = lambda c: run_train(c)["best_val_acc"]`, writes `outputs/tuning/{method}.json`.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: grid and random search with tuner`.

### Task 10: Grad-CAM, evaluation, predict

**Files:**
- Create: `src/visualization/gradcam.py`, `src/evaluation/evaluate.py`, `tests/test_gradcam.py`, `tests/test_evaluate.py`
- Modify: `src/cli.py` (`predict`, `evaluate`, `gradcam` subcommands)

**Interfaces:**
- Consumes: `load_checkpoint` (Task 7), `Encoder.gradcam_layer` (Task 5), `build_transforms` (Task 3).
- Produces: `class GradCAM` with `__init__(model: Classifier, target_layer: nn.Module)`, `__call__(x: Tensor[1,3,H,W], class_idx: int | None = None) -> np.ndarray[H,W]` scaled to [0,1], `close() -> None` (removes hooks); `overlay(image: PIL.Image, cam: np.ndarray, alpha: float = 0.4) -> PIL.Image`; `evaluate(model, loader, device: str) -> dict` with `y_true, y_pred, logits`; `select_gradcam_examples(y_true, y_pred, probs, k: int = 4) -> dict[str, list[int]]` with keys `confident_correct, barely_correct, confident_wrong, confused_pair`; `predict_image(path: str, checkpoint: str, topk: int = 5, device: str = "cpu") -> list[tuple[str, float]]`.

- [ ] **Step 1: Write failing tests** (random-weight `scratch_cnn`, 64x64 inputs): `test_cam_shape_and_range` (output `(H, W)`, min >= 0, max <= 1); `test_cam_differs_by_class`; `test_close_removes_hooks` (no forward/backward hooks remain on the target layer); `test_predict_image_handles_rgba_and_gray` (saves an RGBA PNG and a grayscale JPG, returns `topk` pairs of (name, prob) summing to <= 1); `test_predict_missing_file_clear_error` (`FileNotFoundError` mentioning the path); `test_select_examples_categories` (synthetic probs produce non-overlapping index lists for correct vs wrong categories).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement Grad-CAM by hooking activations and gradients on the target layer: weights = global-average-pooled gradients, `cam = ReLU(sum(weights * activations))`, upsample to input size, min-max normalise. `predict_image` loads the checkpoint on CPU (`map_location`), converts to RGB, applies the eval transform, softmax top-k with class names from the checkpoint. CLI `gradcam` writes a grid figure for the four example categories to `outputs/figures/`.
- [ ] **Step 4:** Run; expected PASS. **Step 5:** Commit `feat: Grad-CAM, evaluation and predict commands`.

### Task 11: Comparison figures and benchmark table

**Files:**
- Create: `src/visualization/comparison.py`, `src/evaluation/benchmark.py`, `configs/benchmark.yaml`, `tests/test_benchmark.py`

**Interfaces:**
- Produces: `plot_model_comparison(results: dict[str, dict], out: Path) -> None` (bars of test accuracy and macro-F1 per run); `plot_training_curves(histories: dict[str, list[dict]], out: Path) -> None`; `plot_confused_pairs(cm, class_names, out: Path, k: int = 10) -> None`; `plot_per_class_f1(y_true, y_pred, class_names, out: Path) -> None`; `benchmark_table(ours: dict[str, float], published: list[dict]) -> pandas.DataFrame`, where each published entry has `name, accuracy, source_url, setup` and a missing or empty `source_url` raises `ValueError`.

- [ ] **Step 1: Write failing tests.** `test_benchmark_table_rows` (ours plus two published entries gives three rows with columns `name, accuracy, source_url, setup`); `test_benchmark_rejects_missing_source`.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement; plotting functions save a PNG and return without showing. **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Look up published CUB-200-2011 classification accuracies (same official train/test split, top-1 accuracy), record each with its source URL and setup (backbone, input size, pretraining data) in `configs/benchmark.yaml`; none are written from memory.
- [ ] **Step 6:** Commit `feat: comparison plots and benchmark table`.

### Task 12: Run the experiments

**Files:**
- Create: `notebooks/colab_runner.ipynb` (only shell cells that clone, install, log in, and call the CLI; no logic)
- Modify: `configs/*.yaml` (epochs and lr after timing), `outputs/NOTES.md`

- [ ] **Step 1:** Local CPU smoke test: `python -m src.cli train --task classification --config configs/scratch_cnn.yaml --smoke`; expected: finishes, writes a checkpoint, no tracker login needed.
- [ ] **Step 2:** On Colab/Kaggle T4: clone the repo, `pip install -r requirements.txt`, `wandb login`, run `python -m src.cli eda`. Measure seconds per epoch for ResNet-18 and set final `epochs`/`patience` so all runs fit; record the measurement in NOTES.md.
- [ ] **Step 3:** Run the 8 training runs, one `train` command per config; tracker group `m1-runs`. Members may split runs across their Colab accounts; each commits its own config or NOTES changes.
- [ ] **Step 4:** Run `tune --method grid` and `tune --method random`; then retrain the best setting once at the full budget.
- [ ] **Step 5:** Evaluate every final checkpoint on the test set (`evaluate`), generate Grad-CAM figures for the best fine-tuned model and the ScratchCNN, and copy checkpoints to Drive.
- [ ] **Step 6:** For anything that failed or underperformed, add what was tried, what happened and the suspected reason to NOTES.md. Commit `docs: experiment notes and final configs`.

### Task 13: README and M1 handoff

**Files:**
- Modify: `README.md`; Create: `outputs/figures/` content produced in Task 12

- [ ] **Step 1:** README: setup (venv, `pip install -r requirements.txt`, `wandb login`), exact commands for `eda`, `train` (all 8 configs), `tune` (both methods), `evaluate`, `gradcam`, `predict`, a Colab/Kaggle recipe, and the note that colorization and caption are Milestone 2.
- [ ] **Step 2:** Verify reproducibility: in a fresh venv, follow the README commands for `train --smoke` and `predict` on one image with a downloaded checkpoint; expected: both run without edits.
- [ ] **Step 3:** Check the spec coverage list (spec section 9) against what exists; list any gap in NOTES.md. Commit `docs: README with setup and run commands`.

---

## Self-review (completed)

- **Spec coverage:** Section 3 -> Tasks 3, 4; section 4 -> Tasks 5, 8; section 5 -> Task 7; section 6 -> Task 9; section 7 -> Tasks 6, 10, 11; section 8 -> Tasks 1, 8, 13; section 9 checklist -> Tasks 12, 13; section 10 order replaced by the single ordered batch above, as requested.
- **Types:** `Encoder`, `Classifier`, `Tracker`, `load_checkpoint`, `run_train`, `TuneResult` signatures are used identically across tasks.
- **Proportion:** tests and signatures only; bodies are left to the implementer except where the spec pins values.
