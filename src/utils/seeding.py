"""Deterministic seeding across numpy / torch / python (SPEC §7.4: seeds fixed in configs)."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
