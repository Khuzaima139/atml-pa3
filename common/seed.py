"""Seeding for python, numpy and (if installed) torch."""
import os
import random

import numpy as np

DEFAULT_SEED = 6304


def seed_everything(seed=DEFAULT_SEED):
    """Seed python, numpy and torch (when importable). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def torch_generator(seed=DEFAULT_SEED):
    """A seeded torch.Generator for DataLoader shuffling."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive worker seeds from torch's initial seed."""
    import torch
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)
