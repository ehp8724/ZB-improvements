"""Performance metrics.

Paper metrics: final Accumulated Portfolio Value (``fapv``), Sharpe ratio
(per-period, risk-free 0; eq. 28) and Maximum Drawdown (eq. 29). Everything else
is the extended set requested for this study.

Annualisation
-------------
Crypto trades continuously. With 30-minute bars there are

    periods_per_year = 365.25 days * 24 h * 2 = 17_532

periods in a year. Returns are annualised geometrically, volatility by
``sqrt(periods_per_year)``. The factor is centralised in ``PERIODS_PER_YEAR`` and
documented in ``docs/methodology.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365.25 * 24 * 2  # 30-minute bars, 24/7 market  -> 17532.0
PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = 48 * 7


def _simple_returns_from_wealth(wealth: np.ndarray) -> np.ndarray:
    wealth = np.asarray(wealth, dtype=np.float64)
    prev = np.concatenate([[1.0], wealth[:-1]])
    return wealth / prev - 1.0


def max_drawdown(wealth: np.ndarray) -> float:
    wealth = np.asarray(wealth, dtype=np.float64)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))
    dd = 1.0 - np.concatenate([[1.0], wealth]) / peak
    return float(dd.max())


def sharpe_ratio(returns: np.ndarray, *, annualise: bool = False, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64) - rf
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    s = r.mean() / sd
    return float(s * np.sqrt(PERIODS_PER_YEAR)) if annualise else float(s)


def sortino_ratio(returns: np.ndarray, *, annualise: bool = False, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64) - rf
    downside = r[r < 0]
    dd = np.sqrt(np.mean(downside**2)) if downside.size else 0.0
    if dd == 0:
        return 0.0
    s = r.mean() / dd
    return float(s * np.sqrt(PERIODS_PER_YEAR)) if annualise else float(s)


def summarise(
    wealth: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    cost: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    gross_wealth: np.ndarray | None = None,
) -> dict:
    """One-shot metric dictionary for a single equity curve."""
    wealth = np.asarray(wealth, dtype=np.float64)
    n = wealth.size
    r = _simple_returns_from_wealth(wealth)
    logr = np.log1p(r)
    years = n / PERIODS_PER_YEAR
    fapv = float(wealth[-1])

    out = {
        "n_periods": int(n),
        "years": float(years),
        "fapv": fapv,
        "total_return": fapv - 1.0,
        "annual_return": float(fapv ** (1.0 / years) - 1.0) if years > 0 else np.nan,
        "annual_vol": float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe_period": sharpe_ratio(r),
        "sharpe_annual": sharpe_ratio(r, annualise=True),
        "sortino_annual": sortino_ratio(r, annualise=True),
        "max_drawdown": max_drawdown(wealth),
        "mean_log_return": float(logr.mean()),
    }
    out["calmar"] = (
        out["annual_return"] / out["max_drawdown"] if out["max_drawdown"] > 0 else np.nan
    )

    if turnover is not None:
        turnover = np.asarray(turnover, dtype=np.float64)
        out["turnover_mean"] = float(turnover.mean())  # one-way, per period
        out["turnover_annual"] = float(turnover.mean() * PERIODS_PER_YEAR)
    if cost is not None:
        cost = np.asarray(cost, dtype=np.float64)
        out["total_cost"] = float(cost.sum())
        out["cost_drag_annual"] = float(cost.sum() / max(years, 1e-9))
    if gross_wealth is not None:
        gross_wealth = np.asarray(gross_wealth, dtype=np.float64)
        out["gross_fapv"] = float(gross_wealth[-1])
        out["cost_fapv_gap"] = float(gross_wealth[-1] - fapv)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        out["avg_cash_weight"] = float(weights[:, 0].mean())
        out["avg_hhi"] = float((weights**2).sum(axis=1).mean())
        out["avg_max_weight"] = float(weights.max(axis=1).mean())

    # worst rolling day / week (product of gross returns over the window)
    if n >= PERIODS_PER_DAY:
        out["worst_day"] = float(_min_rolling_return(r, PERIODS_PER_DAY))
    if n >= PERIODS_PER_WEEK:
        out["worst_week"] = float(_min_rolling_return(r, PERIODS_PER_WEEK))

    if timestamps is not None:
        out["by_year"] = performance_by_year(wealth, timestamps)
    out["by_vol_regime"] = performance_by_vol_regime(r)
    return out


def _min_rolling_return(r: np.ndarray, w: int) -> float:
    g = np.log1p(r)
    csum = np.concatenate([[0.0], np.cumsum(g)])
    roll = csum[w:] - csum[:-w]
    return float(np.expm1(roll.min()))


def performance_by_year(wealth: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    s = pd.Series(wealth, index=pd.DatetimeIndex(timestamps))
    r = s / s.shift(1).fillna(1.0) - 1.0
    out = {}
    for year, grp in r.groupby(r.index.year):
        w = np.cumprod(1.0 + grp.to_numpy())
        out[int(year)] = {
            "return": float(w[-1] - 1.0),
            "sharpe_annual": sharpe_ratio(grp.to_numpy(), annualise=True),
            "max_drawdown": max_drawdown(w),
            "n_periods": int(grp.size),
        }
    return out


def performance_by_vol_regime(r: np.ndarray, window: int = PERIODS_PER_DAY) -> dict:
    r = np.asarray(r, dtype=np.float64)
    if r.size < 2 * window:
        return {}
    roll = pd.Series(r).rolling(window).std().to_numpy()
    med = np.nanmedian(roll)
    hi = roll > med
    lo = ~hi & ~np.isnan(roll)
    out = {}
    for name, mask in (("low_vol", lo), ("high_vol", hi)):
        rr = r[mask]
        if rr.size:
            out[name] = {
                "mean_period_return": float(rr.mean()),
                "sharpe_annual": sharpe_ratio(rr, annualise=True),
                "n_periods": int(rr.size),
            }
    return out


def aggregate_seeds(metric_dicts: list[dict], keys: list[str] | None = None) -> dict:
    """mean / std / median / min / max plus the raw per-seed values."""
    if not metric_dicts:
        return {}
    keys = keys or [
        k
        for k, v in metric_dicts[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out = {}
    for k in keys:
        vals = np.array([float(d[k]) for d in metric_dicts if k in d and np.isfinite(d[k])])
        if vals.size == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": vals.tolist(),
        }
    return out


def block_bootstrap_ci(
    returns: np.ndarray,
    stat: str = "fapv",
    *,
    block: int = PERIODS_PER_DAY,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Circular block-bootstrap CI for a path statistic on per-period simple returns."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    rng = np.random.default_rng(seed)
    ext = np.concatenate([r, r[:block]])

    def compute(sample: np.ndarray) -> float:
        w = np.cumprod(1.0 + sample)
        if stat == "fapv":
            return float(w[-1])
        if stat == "sharpe_annual":
            return sharpe_ratio(sample, annualise=True)
        if stat == "max_drawdown":
            return max_drawdown(w)
        raise ValueError(stat)

    n_blocks = int(np.ceil(n / block))
    est = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        est[b] = compute(ext[idx % len(ext)])
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return {"point": compute(r), "lo": float(lo), "hi": float(hi), "n_boot": n_boot, "block": block}
