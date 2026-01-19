"""
iwae (importance weighted auto-encoder) bounds computation for propensity bound evaluation.

this module implements functions to compute iwae bounds and expectations from importance weights.
"""

from typing import Dict, List

import numpy as np


def iwae_curve(
    logw: np.ndarray,
    ks: List[int],
    resamples: int = 100,
    seed: int | None = None,
) -> Dict[int, float]:
    """
    compute iwae bound curve for different numbers of importance samples.

    the iwae bound is computed as:
    iwae_k = e[log (1/k) * sum_{i=1}^k exp(w_i)]

    where w_i are the log importance weights.

    args:
        logw: array of log importance weights (shape: [n_samples])
        ks: list of k values (number of importance samples to use)
        resamples: number of monte carlo resamples for estimation
        seed: random seed for reproducibility

    returns:
        dictionary mapping k -> iwae bound estimate
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(logw)
    curve = {}

    for k in ks:
        if k > n_samples:
            continue

        # for each resample, randomly select k samples and compute iwae bound
        bounds = []
        for _ in range(resamples):
            # randomly select k indices without replacement
            indices = np.random.choice(n_samples, size=k, replace=False)
            selected_logw = logw[indices]

            # compute iwae bound: log(1/k * sum(exp(w_i)))
            # = log(sum(exp(w_i))) - log(k)
            log_sum_exp = np.logaddexp.reduce(selected_logw)
            bound = log_sum_exp - np.log(k)
            bounds.append(bound)

        # average over resamples
        curve[k] = float(np.mean(bounds))

    return curve


def prbo_expectation(logw: np.ndarray) -> float:
    """
    compute the prbo (propensity bound) expectation.

    this computes e[exp(w)] where w are the log importance weights,
    which gives the expected value under the prbo distribution.

    args:
        logw: array of log importance weights

    returns:
        expected value under prbo distribution
    """
    # prbo expectation is e[exp(w)] = (1/n) * sum(exp(w_i))
    return float(np.mean(np.exp(logw)))


