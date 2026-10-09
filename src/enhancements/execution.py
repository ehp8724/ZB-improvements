"""Dollar-accounting backtest engine with realistic execution (enhancement 1).

The paper assumes every rebalance fills instantly at the decision-bar close and
only pays a flat linear commission. This engine keeps that as ``mode="instant"``
(the paper-faithful baseline) and adds execution algorithms that work the order
over the *following* 30 minutes of 1-minute data:

* ``twap``  equal slices
* ``vwap``  slices proportional to the trailing 7-day same-time-of-day volume profile
* ``is``    implementation shortfall (Almgren-Chriss front-loaded trajectory)

Every algorithm pays the same flat linear ``fee`` (25 bp default, as in the paper)
*plus* half the spread and a square-root market-impact slippage on each child fill:

    slip_k = half_spread_i + impact_coef * sigma_1m,i * sqrt(participation_k)

where ``participation_k`` is child notional / that minute's traded USDT volume and
``sigma_1m`` is the trailing 6-hour minute volatility. Spread/impact levels are
explicit scenario assumptions: candle archives contain no quotes.

Positions are tracked in units with a cash account, so signed (long/short)
portfolios, borrow costs and a per-asset P&L decomposition are exact:
sum over assets of ``asset_pnl`` plus borrow equals the change in equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import BAR, Bars, Market

# half-spread in bp (scenario assumption; Binance spot books are tight for majors)
HALF_SPREAD_BP = {
    "BTC": 0.1, "ETH": 0.2, "BNB": 0.5, "SOL": 0.5, "XRP": 0.5,
    "ADA": 1.0, "DOGE": 1.0, "AVAX": 1.5, "LINK": 1.5, "LTC": 1.5,
}


@dataclass
class ExecConfig:
    mode: str = "instant"  # instant | twap | vwap | is
    fee: float = 0.0025  # flat linear commission on traded notional (paper: 25 bp)
    aum: float = 1_000_000.0  # book size in USDT
    impact_coef: float = 1.0  # sqrt-law coefficient Y
    spread_scale: float = 1.0
    urgency: float = 3.0  # IS: kappa*N (0 -> TWAP, larger -> more front-loaded)
    borrow_apr: float = 0.08  # annual borrow cost on short notional
    vol_lookback: int = 360  # minutes for sigma_1m
    profile_days: int = 7
    include_slippage: bool = True  # False => commission only (for ablation)


@dataclass
class SimResult:
    equity: np.ndarray  # (T+1,) equity at bar boundaries, equity[0]=aum
    ret: np.ndarray  # (T,) net simple return per bar
    gross_ret: np.ndarray  # (T,) return before fee/slippage/borrow
    fee: np.ndarray  # (T,) USDT commission
    slip: np.ndarray  # (T,) USDT implementation shortfall vs the decision close (spread + impact + delay)
    borrow: np.ndarray  # (T,) USDT borrow cost
    turnover: np.ndarray  # (T,) traded notional / equity (one-sided sum |dw|)
    w_pre: np.ndarray  # (T, m) weights just before each rebalance (post-drift)
    w_post: np.ndarray  # (T, m) realised target weights after rebalance (at decision price)
    asset_pnl: np.ndarray  # (T, m) USDT P&L by asset, net of its fee/slippage
    asset_cost: np.ndarray  # (T, m) USDT fee+slippage by asset
    times: pd.DatetimeIndex = field(default=None)  # decision times (T,)
    meta: dict = field(default_factory=dict)

    @property
    def wealth(self) -> np.ndarray:
        return self.equity[1:] / self.equity[0]


def _rolling_sigma(close: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing 1-minute log-return std known *before* each minute (shifted by one)."""
    r = np.diff(np.log(close), axis=0, prepend=np.log(close[:1]))
    s = pd.DataFrame(r).rolling(lookback, min_periods=30).std().shift(1)
    return s.bfill().fillna(1e-4).to_numpy()


def schedule(mode: str, n: int, urgency: float, profile: np.ndarray | None) -> np.ndarray:
    """Child-order fractions over ``n`` minutes, shape (n,) or (n, m) summing to 1."""
    if mode == "twap" or (mode == "is" and urgency <= 1e-9):
        return np.full(n, 1.0 / n)
    if mode == "is":
        k = urgency / n
        x = np.sinh(k * (n - np.arange(n + 1))) / np.sinh(k * n)  # remaining fraction
        return x[:-1] - x[1:]
    if mode == "vwap":
        if profile is None:
            return np.full(n, 1.0 / n)
        p = profile + 1e-9
        return p / p.sum(axis=0, keepdims=True)
    raise ValueError(mode)


class Simulator:
    """Holds precomputed minute-level helpers for repeated backtests."""

    def __init__(self, mkt: Market, bars: Bars):
        self.mkt, self.bars = mkt, bars
        self.sigma = _rolling_sigma(mkt.close, 360)
        self._sigma_lb = 360

    def _sigma_for(self, lookback: int) -> np.ndarray:
        if lookback != self._sigma_lb:
            self.sigma = _rolling_sigma(self.mkt.close, lookback)
            self._sigma_lb = lookback
        return self.sigma

    def run(self, W: np.ndarray, a: int, cfg: ExecConfig) -> SimResult:
        """Simulate decisions at bars ``a .. a+T-1``.

        ``W`` is (T, m) *risky* target weights (signed allowed; cash = 1 - sum).
        Decision ``i`` happens at the close of bar ``a+i``. The first row of the
        book is all cash with equity ``cfg.aum``.
        """
        mk, bars = self.mkt, self.bars
        T, m = W.shape
        assert m == mk.m
        close_b = bars.close
        hs = np.array([HALF_SPREAD_BP[s] for s in mk.symbols]) * 1e-4 * cfg.spread_scale
        sigma = self._sigma_for(cfg.vol_lookback)
        u = np.zeros(m)
        cash = cfg.aum
        E = cfg.aum
        out = {k: np.zeros(T) for k in ("ret", "gross_ret", "fee", "slip", "borrow", "turnover")}
        w_pre = np.zeros((T, m)); w_post = np.zeros((T, m))
        apnl = np.zeros((T, m)); acost = np.zeros((T, m))
        eq = np.empty(T + 1); eq[0] = E
        borrow_rate = cfg.borrow_apr / (365 * 48)
        sched_const = None
        if cfg.mode in ("twap", "is"):
            sched_const = schedule(cfg.mode, BAR, cfg.urgency, None)[:, None]
        for i in range(T):
            b = a + i
            Cd = close_b[b]  # decision price
            Cn = close_b[b + 1] if b + 1 < close_b.shape[0] else Cd
            w_pre[i] = u * Cd / E
            tgt_u = W[i] * E / Cd
            du = tgt_u - u
            s0 = int(bars.minute_start[b]) + BAR  # first minute after decision
            if cfg.mode == "instant" or s0 + BAR > mk.close.shape[0]:
                notional = np.abs(du) * Cd
                fee = cfg.fee * notional
                cash_flow = -du * Cd
                slip_i = np.zeros(m)
                u_new = tgt_u
                cost_i = fee
            else:
                rows = slice(s0, s0 + BAR)
                if cfg.mode == "vwap":
                    prof = np.zeros((BAR, m)); cnt = 0
                    for d in range(1, cfg.profile_days + 1):
                        r0 = s0 - 1440 * d
                        if r0 < 0:
                            break
                        prof += mk.qvol[r0 : r0 + BAR]; cnt += 1
                    sc = schedule("vwap", BAR, 0, prof if cnt else None)
                    if sc.ndim == 1:
                        sc = sc[:, None]
                else:
                    sc = sched_const
                q = sc * du[None, :]  # (BAR, m) signed child quantities
                vw = mk.vwap[rows]
                notional_k = np.abs(q) * vw
                part = np.minimum(notional_k / (mk.qvol[rows] + 1.0), 1.0)
                slip_frac = hs[None, :] + cfg.impact_coef * sigma[rows] * np.sqrt(part)
                if not cfg.include_slippage:
                    slip_frac = np.zeros_like(slip_frac)
                fill = vw * (1.0 + np.sign(q) * slip_frac)
                fee = cfg.fee * notional_k.sum(axis=0)
                cash_flow = -(q * fill).sum(axis=0)
                # implementation shortfall vs the decision-bar close (spread + impact + delay drift)
                slip_i = (q * (fill - Cd[None, :])).sum(axis=0)
                u_new = tgt_u
                cost_i = fee + slip_i
            cash = cash + cash_flow.sum() - fee.sum()
            # per-asset P&L over the bar (equity is marked at the decision close each bar)
            apnl[i] = (u_new * Cn - u * Cd) + cash_flow - fee
            # short borrow on notional carried through the bar
            short_notional = np.maximum(-u_new, 0.0) * Cn
            bcost = borrow_rate * short_notional.sum()
            cash -= bcost
            u = u_new
            E_new = cash + (u * Cn).sum()
            out["fee"][i] = fee.sum()
            out["slip"][i] = slip_i.sum()
            out["borrow"][i] = bcost
            out["turnover"][i] = np.abs(du * Cd).sum() / E
            out["ret"][i] = E_new / E - 1.0
            out["gross_ret"][i] = (E_new + fee.sum() + slip_i.sum() + bcost) / E - 1.0
            w_post[i] = W[i]
            acost[i] = cost_i
            E = E_new
            eq[i + 1] = E
        times = bars.time[a : a + T]
        res = SimResult(eq, out["ret"], out["gross_ret"], out["fee"], out["slip"], out["borrow"],
                        out["turnover"], w_pre, w_post, apnl, acost, times,
                        {"mode": cfg.mode, "aum": cfg.aum})
        return res
