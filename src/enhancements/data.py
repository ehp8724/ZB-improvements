"""Binance 1-minute data -> 30-minute EIIE panel + 1-minute execution grid.

``data/<SYMBOL>/<YYYY-MM>.parquet`` (UTC minute-open index) is aggregated to
30-minute bars labelled by their *close* time, so the bar stamped ``t`` contains
only minutes ``t-29 .. t`` and a decision at ``t`` uses information up to ``t``.
The minute grid is kept so execution algorithms can replay fills inside the
30 minutes *following* a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]
BAR = 30  # minutes per decision bar


@dataclass
class Market:
    """Aligned minute-level market data. Row ``k`` is minute ``index[k]`` (open time)."""

    symbols: list[str]
    index: pd.DatetimeIndex  # minute open times, UTC, complete grid
    close: np.ndarray  # (M, m) float64, forward filled
    high: np.ndarray
    low: np.ndarray
    qvol: np.ndarray  # (M, m) USDT traded per minute (0 when no trades)
    vwap: np.ndarray  # (M, m) minute VWAP = quote_volume/volume, close if no volume

    @property
    def m(self) -> int:
        return len(self.symbols)

    # ---- 30-minute decision grid -------------------------------------------------
    def bars(self) -> "Bars":
        M = self.index.size
        nb = M // BAR
        sl = slice(0, nb * BAR)
        c = self.close[sl].reshape(nb, BAR, self.m)
        h = self.high[sl].reshape(nb, BAR, self.m)
        lo = self.low[sl].reshape(nb, BAR, self.m)
        q = self.qvol[sl].reshape(nb, BAR, self.m)
        # decision time = close of the bar = open time of the next minute
        t_dec = self.index[sl][BAR - 1 :: BAR] + pd.Timedelta(minutes=1)
        return Bars(
            symbols=self.symbols,
            time=pd.DatetimeIndex(t_dec),
            close=c[:, -1, :],
            high=h.max(axis=1),
            low=lo.min(axis=1),
            qvol=q.sum(axis=1),
            minute_start=np.arange(nb) * BAR,
        )


@dataclass
class Bars:
    symbols: list[str]
    time: pd.DatetimeIndex  # decision (bar close) time
    close: np.ndarray  # (T, m)
    high: np.ndarray
    low: np.ndarray
    qvol: np.ndarray  # (T, m) USDT volume inside the bar
    minute_start: np.ndarray  # (T,) first minute-row of each bar in the Market arrays


def load_symbol(root: Path, sym: str, start: str | None, end: str | None) -> pd.DataFrame:
    files = sorted((Path(root) / f"{sym}USDT").glob("*.parquet"))
    frames = []
    for f in files:
        mth = f.stem
        if start and mth < start[:7]:
            continue
        if end and mth > end[:7]:
            continue
        frames.append(pd.read_parquet(f, columns=["open", "high", "low", "close", "volume", "quote_volume"]))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated()]


def load_market(
    root: str | Path = "data",
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    cache: str | Path | None = "artifacts/cache/market.npz",
) -> Market:
    """Load all minute data onto a common complete minute grid (cached as npz)."""
    symbols = symbols or SYMBOLS
    cache = Path(cache) if cache else None
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["symbols"]) == symbols and str(z["start"]) == str(start) and str(z["end"]) == str(end):
            idx = pd.DatetimeIndex(z["index"], tz="UTC")
            return Market(symbols, idx, z["close"], z["high"], z["low"], z["qvol"], z["vwap"])
    dfs = {s: load_symbol(Path(root), s, start, end) for s in symbols}
    lo = max(d.index.min() for d in dfs.values())
    hi = min(d.index.max() for d in dfs.values())
    hi = hi.floor("30min") + pd.Timedelta(minutes=29)  # whole 30-minute bars only
    if hi > min(d.index.max() for d in dfs.values()):
        hi -= pd.Timedelta(minutes=30)
    idx = pd.date_range(lo.ceil("30min"), hi, freq="1min", tz="UTC")
    cols = {k: np.empty((idx.size, len(symbols))) for k in ("close", "high", "low", "qvol", "vwap")}
    for j, s in enumerate(symbols):
        d = dfs[s].reindex(idx)
        vol = d["volume"].fillna(0.0).to_numpy()
        qv = d["quote_volume"].fillna(0.0).to_numpy()
        close = d["close"].ffill().bfill().to_numpy()
        cols["close"][:, j] = close
        cols["high"][:, j] = d["high"].fillna(pd.Series(close, index=idx)).to_numpy()
        cols["low"][:, j] = d["low"].fillna(pd.Series(close, index=idx)).to_numpy()
        cols["qvol"][:, j] = qv
        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.where(vol > 0, qv / np.where(vol > 0, vol, 1.0), close)
        cols["vwap"][:, j] = vw
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, symbols=np.array(symbols), start=str(start), end=str(end),
                 index=idx.tz_convert(None).values, **cols)
    return Market(symbols, idx, cols["close"], cols["high"], cols["low"], cols["qvol"], cols["vwap"])
