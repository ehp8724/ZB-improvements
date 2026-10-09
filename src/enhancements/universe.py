"""Periodic liquidity-based universe selection (enhancement 4).

The paper fixes the asset set (and admits choosing it with look-ahead). Here the
set is re-ranked every ``freq`` using only *trailing* USDT volume, so a coin that
stops trading drops out and a newly liquid one enters. The EIIE evaluator is
shared across assets, so a K-slot model applies to any K-subset unchanged; what
changes is the weight-transition problem, which is charged through the normal
cost model (a departing coin must be sold, an entering one bought).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Bars


def rank_universe(bars: Bars, k: int, freq: str = "MS", lookback_bars: int = 30 * 48) -> np.ndarray:
    """Return (T, K) sorted asset ids per decision bar, constant within each period.

    The ranking at a period start uses volume of the ``lookback_bars`` bars that end at
    or before that bar (no look-ahead). Before enough history exists the full-sample
    first ranking is not used: the first available ranking applies.
    """
    T, m = bars.qvol.shape
    cs = np.vstack([np.zeros((1, m)), np.cumsum(bars.qvol, axis=0)])
    period = bars.time.tz_localize(None).to_period("M" if freq in ("MS", "M") else freq)
    first = np.r_[True, period[1:] != period[:-1]]
    starts = np.flatnonzero(first)
    A = np.zeros((T, k), dtype=np.int64)
    cur = None
    s_iter = set(starts.tolist())
    for t in range(T):
        if t in s_iter or cur is None:
            lo = max(0, t + 1 - lookback_bars)
            vol = cs[t + 1] - cs[lo]
            cur = np.sort(np.argsort(-vol)[:k])
        A[t] = cur
    return A


def fixed_universe(bars: Bars, k: int, at_bar: int, lookback_bars: int = 30 * 48) -> np.ndarray:
    """Static top-K set chosen once from the ``lookback_bars`` ending at ``at_bar``."""
    lo = max(0, at_bar - lookback_bars)
    vol = bars.qvol[lo:at_bar].sum(axis=0)
    ids = np.sort(np.argsort(-vol)[:k])
    return np.tile(ids, (bars.qvol.shape[0], 1))


def membership_changes(A: np.ndarray) -> pd.DataFrame:
    """Dates where the active set changes: entering / leaving asset ids."""
    rows = []
    for t in range(1, A.shape[0]):
        if not np.array_equal(A[t], A[t - 1]):
            rows.append({"bar": t, "enter": sorted(set(A[t]) - set(A[t - 1])),
                         "leave": sorted(set(A[t - 1]) - set(A[t]))})
    return pd.DataFrame(rows)
