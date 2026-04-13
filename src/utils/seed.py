from __future__ import annotations

import os
import random

import numpy as np
import torch
import pyro


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Covers:
    - Python random
    - NumPy
    - PyTorch (CPU + CUDA)
    - Pyro
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)