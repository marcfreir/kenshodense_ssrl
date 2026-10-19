"""
paths.py
Single source of truth for dataset locations and the output directory
layout shared by pretrain.py, finetune.py and evaluate.py.

Defaults can be overridden with environment variables or with the
``--data-root`` / ``--output-root`` command-line flags of each script:

    KENSHODENSE_DATA_ROOT    dataset root (contains images/ and annotations/)
    KENSHODENSE_OUTPUT_ROOT  where checkpoints, logs and results are written

Output layout (k = backbone width multiplier, r = run index):

    <output_root>/
    ├── checkpoints/
    │   ├── pretrain/rn50x{k}/run{r}/   KenshoDense checkpoints
    │   └── finetune/rn50x{k}/run{r}/   DeepLabV3 checkpoints
    ├── logs/
    │   ├── pretrain/                   Lightning CSV logs
    │   └── finetune/
    ├── results/
    │   ├── pretrain/rn50x{k}/          per-run metadata, losses, summaries
    │   └── finetune/rn50x{k}/          per-run finetuning summaries
    └── evaluation/rn50x{k}/run{r}/     test metrics per checkpoint
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("KENSHODENSE_DATA_ROOT", "data/f3_segmentation"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("KENSHODENSE_OUTPUT_ROOT", "outputs"))

# F3 Netherlands inline sections are padded/resized to this (H, W).
IMG_SIZE = (255, 701)
NUM_CLASSES = 6


def images_dir(data_root: Path) -> Path:
    return Path(data_root) / "images"


def annotations_dir(data_root: Path) -> Path:
    return Path(data_root) / "annotations"


def width_tag(width_multiplier: int) -> str:
    return f"rn50x{int(width_multiplier)}"


def pretrain_ckpt_dir(output_root: Path, width_multiplier: int, run_idx: int) -> Path:
    return Path(output_root) / "checkpoints" / "pretrain" / width_tag(width_multiplier) / f"run{run_idx}"


def finetune_ckpt_dir(output_root: Path, width_multiplier: int, run_idx: int) -> Path:
    return Path(output_root) / "checkpoints" / "finetune" / width_tag(width_multiplier) / f"run{run_idx}"


def logs_dir(output_root: Path, stage: str) -> Path:
    return Path(output_root) / "logs" / stage


def results_dir(output_root: Path, stage: str, width_multiplier: int) -> Path:
    return Path(output_root) / "results" / stage / width_tag(width_multiplier)


def evaluation_dir(output_root: Path, width_multiplier: int, run_idx: int) -> Path:
    return Path(output_root) / "evaluation" / width_tag(width_multiplier) / f"run{run_idx}"


def pretrain_model_name(config_key: str, run_idx: int, epochs: int) -> str:
    """Checkpoint stem for one pretraining run, e.g.
    ``kenshodense_GB_param1_run3_100epochs`` (Lightning appends ``_best`` /
    ``_last``). finetune.py parses this format back, so keep them in sync."""
    return f"kenshodense_{config_key}_run{run_idx}_{epochs}epochs"
