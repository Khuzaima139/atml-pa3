"""Paired question-level percentile bootstrap (Task 3, manual L614-615).

Every call with the same seed and sample size draws the same resample indices, so all comparisons share one
set of question resamples.
"""
import numpy as np


def resample_indices(n, n_resamples=1000, seed=6304):
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def paired_bootstrap(a, b, n_resamples=1000, seed=6304, ci=0.95):
    """Mean of a - b over units (questions) and its percentile CI from resampling units with replacement."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    d = a - b
    stats = d[resample_indices(len(d), n_resamples, seed)].mean(axis=1)
    lo, hi = np.percentile(stats, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(d)),
            "resamples": int(n_resamples), "seed": int(seed), "ci": ci}
