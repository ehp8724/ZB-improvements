"""Liquidity-tercile P&L attribution (professor's suggestion, enhancement 5).

Question: is the strategy's P&L / Sharpe concentrated in illiquid ("shitcoin")
names, and is it earned by *trading* (timing) or by simply *holding* them?

No market-cap field exists in the data, so trailing 30-day USDT traded volume is the
size/liquidity proxy. Terciles are re-ranked monthly from trailing data only (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Bars
from .execution import SimResult
from .metrics import PPY

BUCKETS = ["low-liquidity", "mid-liquidity", "high-liquidity"]


def liquidity_buckets(bars: Bars, lookback_bars: int = 30 * 48) -> np.ndarray:
    """(T, m) int in {0,1,2} (0 = least liquid tercile), refreshed on the first bar of each month."""
    T, m = bars.qvol.shape
    cs = np.vstack([np.zeros((1, m)), np.cumsum(bars.qvol, axis=0)])
    per = bars.time.tz_localize(None).to_period("M")
    out = np.zeros((T, m), dtype=np.int64)
    cur = None
    for t in range(T):
        if cur is None or per[t] != per[t - 1]:
            lo = max(0, t + 1 - lookback_bars)
            order = np.argsort(cs[t + 1] - cs[lo])  # ascending liquidity
            cur = np.zeros(m, dtype=np.int64)
            for b, idx in enumerate(np.array_split(order, 3)):
                cur[idx] = b
        out[t] = cur
    return out


def attribute(res: SimResult, bars: Bars, a: int, buckets: np.ndarray) -> pd.DataFrame:
    """Per-tercile decomposition for one backtest starting at decision bar ``a``."""
    T, m = res.w_post.shape
    bk = buckets[a : a + T]
    E_prev = res.equity[:-1]
    C = bars.close
    r_asset = C[a + 1 : a + T + 1] / C[a : a + T] - 1.0  # (T, m) bar return of each asset
    W = res.w_post
    years = T / PPY
    rows = []
    wbar = W.mean(axis=0)
    gross_c = W * r_asset  # gross P&L contribution per bar (fraction of equity)
    passive_i = wbar * r_asset.sum(axis=0)
    active_i = gross_c.sum(axis=0) - passive_i
    tot_pnl = res.asset_pnl.sum()
    for b, name in enumerate(BUCKETS):
        mask = bk == b
        sleeve = (res.asset_pnl * mask).sum(axis=1) / E_prev  # net sleeve return per bar
        pnl = (res.asset_pnl * mask).sum()
        expo = (np.abs(W) * mask).sum(axis=1)
        bench = np.where(mask.sum(1) > 0, (r_asset * mask).sum(1) / np.maximum(mask.sum(1), 1), 0.0)
        # asset sets are time-varying; map static passive/active split by time-weighted bucket share
        share_t = mask.mean(axis=0)  # fraction of bars each asset sits in this bucket
        rows.append({
            "bucket": name,
            "avg_assets": float(mask.sum(1).mean()),
            "avg_gross_weight": float(expo.mean()),
            "net_pnl_usd": float(pnl),
            "share_of_pnl_pct": float(100 * pnl / tot_pnl) if abs(tot_pnl) > 1e-9 else np.nan,
            "ann_contribution_pct": float(sleeve.sum() / years * 100),
            "sleeve_sharpe": float(sleeve.mean() / (sleeve.std() + 1e-15) * np.sqrt(PPY)),
            "cost_pct_of_aum": float((res.asset_cost * mask).sum() / res.equity[0] * 100),
            "turnover_share_pct": float(
                100 * (np.abs(np.diff(np.vstack([np.zeros(m), W]), axis=0)) * mask).sum()
                / max(np.abs(np.diff(np.vstack([np.zeros(m), W]), axis=0)).sum(), 1e-12)),
            "gross_passive_bps_total": float((passive_i * share_t).sum() * 1e4),
            "gross_active_bps_total": float((active_i * share_t).sum() * 1e4),
            "bucket_ew_bh_return_pct": float(((1 + bench).prod() - 1) * 100),
            "return_per_unit_exposure_bps_bar": float(
                (gross_c * mask).sum() / max((np.abs(W) * mask).sum(), 1e-12) * 1e4),
        })
    return pd.DataFrame(rows)
