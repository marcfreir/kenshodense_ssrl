"""
seed_utils.py
Centralized reproducibility helpers shared by the pretrain and finetune
scripts.

Design rationale
----------------
Every augmentation configuration is trained R times (repeated runs) so that
per-configuration statistics (mean/std mIoU, stability) and
cross-configuration paired tests (e.g. the Wilcoxon signed-rank test) can be
computed. For that to be valid:

  1. Each run must be reproducible on its own -- re-running "run3" of a
     given configuration twice must give the same results.
  2. Run r of configuration A and run r of configuration B must be a
     legitimate *pair* for the paired test. That requires holding every
     other source of randomness (weight initialization, data
     split/shuffling order, DataLoader worker seeding, ...) fixed across
     configurations for a given r, so that the transformation being
     tested is the only thing that differs between the two runs in the
     pair.

Both requirements are satisfied by making the seed a function of the run
index ONLY -- never of the transformation/config key. Every configuration's
"run 3" therefore uses the exact same seed, and only the augmentation
pipeline itself differs between configs at a given run index.

Usage
-----
    from seed_utils import seed_everything_for_run

    seed = seed_everything_for_run(run_idx)   # call before building the
                                              # datamodule/model/trainer
"""

import random

import numpy as np
import torch
import lightning as L

# Change this single constant to reseed the entire study (e.g. for a fully
# independent replication with a different base seed).
SEED_BASE = 42


def seed_for_run(run_idx: int) -> int:
    """Deterministic seed for a given repeated-run index (1-based)."""
    return SEED_BASE + int(run_idx)


def seed_everything_for_run(run_idx: int, workers: bool = True) -> int:
    """
    Seed Python's `random`, NumPy, PyTorch (CPU + all CUDA devices), and
    Lightning's own RNG-dependent machinery (DataLoader worker seeding,
    when `workers=True`) for repeated run `run_idx`.

    `lightning.pytorch.seed_everything` already covers `random`, `numpy`
    and `torch` internally; the explicit calls below are belt-and-braces.

    Returns the seed that was set, so callers can record it alongside the
    run's other metadata.
    """
    seed = seed_for_run(run_idx)

    L.seed_everything(seed, workers=workers)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[seed_utils] run_idx={run_idx} -> seed={seed} (SEED_BASE={SEED_BASE})")

    return seed
