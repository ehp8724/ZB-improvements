"""Chronological windowed datasets and leakage-safe splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .price_tensor import build_windows


@dataclass
class WindowedData:
    X: np.ndarray  # (N, 3, m, n)
    y_next: np.ndarray  # (N, m+1)
    t_index: np.ndarray  # (N,) bar index of each decision time
    train: np.ndarray  # bool mask over the N samples
    val: np.ndarray
    test: np.ndarray
    n_assets: int
    window: int

    def slice(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.X[mask], self.y_next[mask], self.t_index[mask]


def make_windowed(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
    train_end_bar: int,
    val_end_bar: int,
) -> WindowedData:
    """Build overlapping states for the whole panel, then assign each decision
    time to train / val / test by its *bar index*.

    ``train_end_bar`` / ``val_end_bar`` are bar indices: a decision time ``t`` is
    training iff ``t < train_end_bar``, validation iff ``train_end_bar <= t <
    val_end_bar``, test otherwise. A sample's state only ever looks back, and its
    label ``y_next`` is the very next bar, so a sample assigned to a split never
    consumes bars from a later split beyond that single realised step.

    Per-window normalisation (divide by the last close) fits no global statistic,
    so no train-only scaler is required; if one is ever added it must be fit on
    ``train`` alone.
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    X, y_next, t_index = build_windows(close, high, low, window)
    m = close.shape[1]

    train = t_index < train_end_bar
    val = (t_index >= train_end_bar) & (t_index < val_end_bar)
    test = t_index >= val_end_bar
    return WindowedData(X, y_next, t_index, train, val, test, m, window)
