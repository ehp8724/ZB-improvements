"""Online Stochastic Batch Learning sampler (arXiv:1706.10059v2, §5.3, eq. 26).

A training batch is a *contiguous* block of ``batch_window`` decision times
starting at ``t_b``. The start is drawn from a geometric law that favours recent
data:

    P_beta(t_b) = beta * (1 - beta) ** ( t_max - t_b )

with ``t_max = n_periods - batch_window`` the most recent admissible start and
``beta`` small. We sample the backward offset ``k = t_max - t_b`` from a geometric
distribution truncated to ``[0, t_max]`` and renormalised on that finite support.
"""

from __future__ import annotations

import numpy as np


def geometric_batch_start(
    n_periods: int,
    batch_window: int,
    beta: float,
    rng: np.random.Generator,
) -> int:
    """Return a batch start index ``t_b`` in ``[0, n_periods - batch_window]``."""
    t_max = n_periods - batch_window
    if t_max < 0:
        raise ValueError(f"batch_window {batch_window} > n_periods {n_periods}")
    if t_max == 0:
        return 0
    # k ~ Geometric(beta) truncated to [0, t_max]; p(k) ∝ (1-beta)^k
    k = rng.geometric(beta) - 1  # numpy geometric is on {1,2,...}
    while k > t_max:
        k = rng.geometric(beta) - 1
    return int(t_max - k)


def batch_indices(t_b: int, batch_window: int) -> np.ndarray:
    """Decision-time indices covered by a batch starting at ``t_b``."""
    return np.arange(t_b, t_b + batch_window)
