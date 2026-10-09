"""Windowed EIIE states built lazily from the 30-minute bars (no X tensor copy).

The state at decision bar ``t`` is the paper's (3, K, n) tensor of close/high/low
over bars ``t-n+1..t`` divided by ``close[t]`` per asset (eq. 18); only past bars
are used. ``K`` assets are gathered through an optional per-bar universe index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import Bars


def split_bars(bars: Bars, train_end: str, val_end: str) -> dict[str, int]:
    """Bar indices of the split boundaries (decision at index t trades bar t+1)."""
    ts = bars.time
    te = int(ts.searchsorted(pd.Timestamp(train_end, tz="UTC")))
    ve = int(ts.searchsorted(pd.Timestamp(val_end, tz="UTC")))
    return {"train_end": te, "val_end": ve, "T": len(ts)}


class Windows:
    def __init__(self, bars: Bars, n: int = 50):
        self.n, self.T, self.m = n, bars.close.shape[0], bars.close.shape[1]
        self.close = torch.as_tensor(bars.close, dtype=torch.float32)
        self.cw = self.close.unfold(0, n, 1)  # (T-n+1, m, n)
        self.hw = torch.as_tensor(bars.high, dtype=torch.float32).unfold(0, n, 1)
        self.lw = torch.as_tensor(bars.low, dtype=torch.float32).unfold(0, n, 1)
        y = np.ones((self.T, self.m + 1))
        y[1:, 1:] = bars.close[1:] / bars.close[:-1]
        self.y = torch.as_tensor(y, dtype=torch.float32)  # y[t] = close[t]/close[t-1], col0 cash
        self.t_min, self.t_max = n - 1, self.T - 2  # decision indices with a next bar

    def state(self, t: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """(B,) decision indices [+ (B,K) asset ids] -> (B, 3, K, n)."""
        i = t - (self.n - 1)
        ref = self.close[t].unsqueeze(-1)  # (B, m, 1)
        x = torch.stack([self.cw[i], self.hw[i], self.lw[i]], dim=1) / ref.unsqueeze(1)
        if A is not None:
            x = torch.take_along_dim(x, A[:, None, :, None], dim=2)
        return x

    def y_next(self, t: torch.Tensor) -> torch.Tensor:
        """Full-universe price relative over the bar after decision ``t``: (B, m+1)."""
        return self.y[t + 1]


def gather_prev(w_full: torch.Tensor, A: torch.Tensor | None, signed: bool = False) -> torch.Tensor:
    """Map a full-universe weight vector (B, m+1; col 0 cash) onto the active set.

    Mass held in assets that left the universe is liquidated to cash (long-only) or,
    when ``signed``, simply dropped from the model input with cash = 1 - sum(risky).
    """
    if A is None:
        return w_full
    risky = torch.take_along_dim(w_full[:, 1:], A, dim=1)
    cash = 1.0 - risky.sum(dim=1, keepdim=True)
    return torch.cat([cash, risky], dim=1)


def scatter_full(w: torch.Tensor, A: torch.Tensor | None, m: int) -> torch.Tensor:
    """Model output (B, K+1) -> full universe (B, m+1); inactive assets get 0."""
    if A is None:
        return w
    out = torch.zeros(w.shape[0], m + 1, dtype=w.dtype)
    out[:, 0] = w[:, 0]
    out.scatter_(1, A + 1, w[:, 1:])
    return out
