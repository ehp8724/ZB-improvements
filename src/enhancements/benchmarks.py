"""Classical benchmarks, expressed as risky target weights so every strategy goes through
the same execution/accounting engine (the paper's baselines: UBAH, UCRP, best stock,
follow-the-winner / follow-the-loser)."""

from __future__ import annotations

import numpy as np


def ubah(close: np.ndarray) -> np.ndarray:
    rel = close / close[0]
    return rel / rel.sum(axis=1, keepdims=True)


def ucrp(T: int, m: int) -> np.ndarray:
    return np.full((T, m), 1.0 / m)


def single(T: int, m: int, j: int) -> np.ndarray:
    w = np.zeros((T, m)); w[:, j] = 1.0
    return w


def best_stock(close: np.ndarray) -> np.ndarray:
    """Hindsight: all-in on the asset with the best buy-and-hold return."""
    j = int(np.argmax(close[-1] / close[0]))
    return single(close.shape[0], close.shape[1], j)


def rank_strategy(close_full: np.ndarray, a: int, T: int, lookback: int, k: int, sign: int,
                  hold: int = 1) -> np.ndarray:
    """Equal-weight the ``k`` best (sign=+1, follow the winner) or worst (sign=-1, follow the
    loser) assets by trailing ``lookback``-bar return; refreshed every ``hold`` bars."""
    m = close_full.shape[1]
    W = np.zeros((T, m))
    cur = np.zeros(m)
    for i in range(T):
        if i % hold == 0:
            t = a + i
            ret = close_full[t] / close_full[t - lookback] - 1.0
            idx = np.argsort(-sign * ret)[:k]
            cur = np.zeros(m); cur[idx] = 1.0 / k
        W[i] = cur
    return W
