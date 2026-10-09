"""Performance summaries for ``SimResult`` (30-minute bars => 17,520 per year)."""

from __future__ import annotations

import numpy as np

from .execution import SimResult

PPY = 48 * 365


def perf(res: SimResult) -> dict:
    r = res.ret
    w = res.equity[1:] / res.equity[0]
    peak = np.maximum.accumulate(np.r_[1.0, w])
    mdd = float((1 - np.r_[1.0, w] / peak).max())
    years = len(r) / PPY
    sd = r.std() + 1e-15
    traded = float((res.turnover * res.equity[:-1]).sum())
    return {
        "slippage_bps_per_traded": float(res.slip.sum() / max(traded, 1e-9) * 1e4),
        "fapv": float(w[-1]),
        "total_return_pct": float((w[-1] - 1) * 100),
        "ann_return_pct": float((w[-1] ** (1 / years) - 1) * 100) if w[-1] > 0 else -100.0,
        "sharpe": float(r.mean() / sd * np.sqrt(PPY)),
        "max_drawdown_pct": mdd * 100,
        "turnover_per_bar": float(res.turnover.mean()),
        "fee_pct_of_aum": float(res.fee.sum() / res.equity[0] * 100),
        "slippage_pct_of_aum": float(res.slip.sum() / res.equity[0] * 100),
        "borrow_pct_of_aum": float(res.borrow.sum() / res.equity[0] * 100),
        "avg_gross": float(np.abs(res.w_post).sum(1).mean()),
        "avg_net": float(res.w_post.sum(1).mean()),
        "gross_fapv": float(np.prod(1 + res.gross_ret)),
        "n_bars": int(len(r)),
    }
