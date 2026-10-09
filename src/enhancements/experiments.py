"""Experiment stages. Every enhancement is compared with the *unchanged* paper baseline
(long-only EIIE-CNN, 10 fixed coins, instant fills at the bar close, flat 25 bp)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl_portfolio.models.eiie import EIIEConfig, build_eiie

from . import benchmarks as bm
from .attribution import attribute, liquidity_buckets
from .data import Bars, Market, load_market
from .dataset import Windows, split_bars
from .execution import ExecConfig, Simulator
from .metrics import perf
from .networks import LongShortEIIE
from .train import PolicyCfg, SACCfg, quick_eval, rollout, train_policy, train_sac
from .universe import fixed_universe, membership_changes, rank_universe

OUT = Path("artifacts/enh")
TRAIN_END, VAL_END = "2024-01-01", "2025-01-01"
BORROW_APR = 0.08


@dataclass
class Ctx:
    mkt: Market
    bars: Bars
    win: Windows
    sp: dict
    sim: Simulator
    t_val: tuple[int, int]
    t_test: tuple[int, int]

    @property
    def a(self) -> int:  # first test decision bar
        return self.t_test[0]

    @property
    def n_test(self) -> int:
        return self.t_test[1] - self.t_test[0] + 1


def make_ctx() -> Ctx:
    mkt = load_market()
    bars = mkt.bars()
    win = Windows(bars)
    sp = split_bars(bars, TRAIN_END, VAL_END)
    return Ctx(mkt, bars, win, sp, Simulator(mkt, bars),
               (sp["train_end"], sp["val_end"] - 1), (sp["val_end"], win.t_max))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- weight store
def save_w(name: str, split: str, W: np.ndarray) -> None:
    d = OUT / "weights"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}_{split}.npz", W=W.astype(np.float32))


def load_w(name: str, split: str = "test") -> np.ndarray:
    return np.load(OUT / "weights" / f"{name}_{split}.npz")["W"].astype(np.float64)


def have_w(name: str) -> bool:
    return (OUT / "weights" / f"{name}_test.npz").exists()


def write_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))


def read_json(name: str):
    return json.loads((OUT / name).read_text())


def run_sim(ctx: Ctx, W_full: np.ndarray, **kw):
    """Backtest full-universe weights (col 0 = cash) over the test period."""
    cfg = ExecConfig(borrow_apr=BORROW_APR, **kw)
    return ctx.sim.run(W_full[:, 1:], ctx.a, cfg)


def select_and_store(ctx, name, models_and_rollouts, c=0.0025, borrow=0.0):
    """Pick the seed with the best validation Sharpe; save its val/test weights."""
    rows = []
    for tag, wv, wt in models_and_rollouts:
        ev = quick_eval(ctx.win, wv, ctx.t_val[0], c, borrow)
        rows.append({"seed": tag, **ev})
        save_w(f"{name}_s{tag}", "val", wv); save_w(f"{name}_s{tag}", "test", wt)
    best = max(rows, key=lambda r: r["sharpe"])
    for tag, wv, wt in models_and_rollouts:
        if tag == best["seed"]:
            save_w(name, "val", wv); save_w(name, "test", wt)
    return rows, best["seed"]


# ---------------------------------------------------------------- stage: baseline
def stage_baseline(ctx: Ctx, seeds=(0, 1, 2), steps=80_000, commission=0.0025, name="eiie"):
    outs = []
    for s in seeds:
        log(f"[{name}] training seed {s}")
        model = build_eiie(EIIEConfig(n_assets=ctx.win.m))
        train_policy(model, ctx.win, None, ctx.win.t_min, ctx.sp["train_end"] - 1,
                     PolicyCfg(steps=steps, seed=s, commission=commission), log)
        wv = rollout(model, ctx.win, None, *ctx.t_val)
        wt = rollout(model, ctx.win, None, *ctx.t_test)
        outs.append((s, wv, wt))
    rows, best = select_and_store(ctx, name, outs)
    write_json(f"{name}_selection.json", {"rows": rows, "selected_seed": best})
    log(f"[{name}] selected seed {best}: {rows}")


# ---------------------------------------------------------------- stage: benchmarks
def stage_benchmarks(ctx: Ctx):
    b, a, T = ctx.bars, ctx.a, ctx.n_test
    close = b.close
    cut = close[a : a + T + 1]
    m = close.shape[1]
    strat = {
        "ubah": bm.ubah(close[a : a + T]),
        "ucrp": bm.ucrp(T, m),
        "btc_hold": bm.single(T, m, 0),
        "best_stock_hindsight": bm.best_stock(cut)[:T],
        "momentum_1d": bm.rank_strategy(close, a, T, 48, 3, +1, hold=48),
        "reversal_3h": bm.rank_strategy(close, a, T, 6, 3, -1, hold=1),
    }
    for k, w in strat.items():
        W = np.hstack([1 - w.sum(1, keepdims=True), w])
        save_w(k, "test", W)
    log("benchmarks saved")


def stage_costblind(ctx: Ctx, seeds=(0,), steps=80_000):
    """EIIE trained without any cost term: an *active* policy used to expose execution costs."""
    stage_baseline(ctx, seeds=seeds, steps=steps, commission=0.0, name="eiie_costblind")


# ---------------------------------------------------------------- enhancement 1: execution
EXEC_STRATS = ["eiie", "ucrp", "ubah", "momentum_1d", "reversal_3h", "eiie_costblind"]


def stage_exec(ctx: Ctx):
    rows = []
    for name in EXEC_STRATS:
        if not have_w(name):
            log(f"skip {name} (no weights)"); continue
        W = load_w(name)
        for aum in (1e6, 1e7):
            for mode in ("instant", "twap", "vwap", "is"):
                res = run_sim(ctx, W, mode=mode, aum=aum)
                rows.append({"strategy": name, "mode": mode, "aum": aum, **perf(res)})
                log(f"{name:15s} {mode:7s} ${aum:,.0f} fapv {rows[-1]['fapv']:.4f} "
                    f"sharpe {rows[-1]['sharpe']:+.2f} slip {rows[-1]['slippage_pct_of_aum']:.2f}%")
    write_json("exec_table.json", rows)
    # sensitivity on the two most active strategies
    sens = []
    for name in ("reversal_3h", "eiie_costblind"):
        if not have_w(name):
            continue
        W = load_w(name)
        for label, kw in [
            ("twap, commission only (isolates 30-min delay)", dict(mode="twap", include_slippage=False)),
            ("twap, Y=0.5", dict(mode="twap", impact_coef=0.5)),
            ("twap, Y=1", dict(mode="twap", impact_coef=1.0)),
            ("twap, Y=2", dict(mode="twap", impact_coef=2.0)),
            ("is, urgency 1", dict(mode="is", urgency=1.0)),
            ("is, urgency 3", dict(mode="is", urgency=3.0)),
            ("is, urgency 6", dict(mode="is", urgency=6.0)),
            ("twap, fee 10bp", dict(mode="twap", fee=0.0010)),
        ]:
            res = run_sim(ctx, W, aum=1e7, **kw)
            sens.append({"strategy": name, "variant": label, **perf(res)})
    write_json("exec_sensitivity.json", sens)
    # low-fee regime: with a 2 bp fee (maker/VIP-like) execution costs are no longer drowned
    # by commission, so the algorithms can be ranked on an active strategy
    low = []
    for name in ("eiie", "momentum_1d", "reversal_3h", "eiie_costblind"):
        if not have_w(name):
            continue
        W = load_w(name)
        for mode in ("instant", "twap", "vwap", "is"):
            for aum in (1e6, 1e7):
                res = run_sim(ctx, W, mode=mode, aum=aum, fee=0.0002)
                low.append({"strategy": name, "mode": mode, "aum": aum, **perf(res)})
    write_json("exec_lowfee.json", low)


# ---------------------------------------------------------------- enhancement 2: SAC
def stage_sac(ctx: Ctx, seeds=(0, 1, 2), steps=300_000):
    outs, hists = [], {}
    for s in seeds:
        log(f"[sac] seed {s}")
        r = train_sac(ctx.win, None, ctx.win.t_min, ctx.sp["train_end"] - 1,
                      SACCfg(total_steps=steps, seed=s), val_range=ctx.t_val, log=log)
        wv = rollout(r["model"], ctx.win, None, *ctx.t_val)
        wt = rollout(r["model"], ctx.win, None, *ctx.t_test)
        outs.append((s, wv, wt)); hists[s] = r["history"]
        d = OUT / "models"; d.mkdir(parents=True, exist_ok=True)
        torch.save(r["model"].state_dict(), d / f"sac_s{s}.pt")
    rows, best = select_and_store(ctx, "sac", outs)
    write_json("sac_selection.json", {"rows": rows, "selected_seed": best, "history": hists})


# ---------------------------------------------------------------- enhancement 3: long / short
LS_VARIANTS = {  # name -> (max_gross, neutral)
    "ls_free_1.0": (1.0, False),
    "ls_free_1.5": (1.5, False),
    "ls_free_3.0": (3.0, False),
    "ls_neutral_1.5": (1.5, True),
}


def stage_ls(ctx: Ctx, seeds=(0, 1, 2), steps=80_000, variants=None):
    for name, (cap, neutral) in LS_VARIANTS.items():
        if variants and name not in variants:
            continue
        outs = []
        for s in seeds:
            log(f"[{name}] seed {s}")
            net = LongShortEIIE(max_gross=cap, neutral=neutral)
            train_policy(net, ctx.win, None, ctx.win.t_min, ctx.sp["train_end"] - 1,
                         PolicyCfg(steps=steps, seed=s, lr=1e-4, borrow_per_bar=BORROW_APR / (48 * 365)), log)
            outs.append((s, rollout(net, ctx.win, None, *ctx.t_val), rollout(net, ctx.win, None, *ctx.t_test)))
        rows, best = select_and_store(ctx, name, outs, borrow=BORROW_APR / (48 * 365))
        write_json(f"{name}_selection.json", {"rows": rows, "selected_seed": best})


# ---------------------------------------------------------------- enhancement 4: universe
K_UNIV = 6


def _universe(ctx, kind, freq="MS"):
    if kind == "fixed":
        return fixed_universe(ctx.bars, K_UNIV, ctx.sp["train_end"])
    return rank_universe(ctx.bars, K_UNIV, freq=freq)


def stage_universe(ctx: Ctx, seeds=(0, 1, 2), steps=80_000, variants=None):
    for kind in (variants or ("fixed", "dyn")):
        A = _universe(ctx, kind)
        if kind == "dyn":
            write_json("universe_changes.json",
                       membership_changes(A).assign(time=lambda d: ctx.bars.time[d.bar].astype(str))
                       .drop(columns="bar").to_dict("records"))
        outs = []
        for s in seeds:
            log(f"[eiie_{kind}{K_UNIV}] seed {s}")
            model = build_eiie(EIIEConfig(n_assets=K_UNIV))
            train_policy(model, ctx.win, A, ctx.win.t_min, ctx.sp["train_end"] - 1,
                         PolicyCfg(steps=steps, seed=s), log)
            d = OUT / "models"; d.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), d / f"eiie_{kind}{K_UNIV}_s{s}.pt")
            outs.append((s, rollout(model, ctx.win, A, *ctx.t_val), rollout(model, ctx.win, A, *ctx.t_test)))
        rows, best = select_and_store(ctx, f"eiie_{kind}{K_UNIV}", outs)
        write_json(f"eiie_{kind}{K_UNIV}_selection.json", {"rows": rows, "selected_seed": best})
        # K-asset benchmarks on the same universes: equal weight over the active set, rebalanced
        a, T = ctx.a, ctx.n_test
        Wc = np.zeros((T, ctx.win.m + 1))
        for i in range(T):
            Wc[i, 1 + A[a + i]] = 1.0 / K_UNIV
        save_w(f"ucrp_{kind}{K_UNIV}", "test", Wc)


def _crash_market(mkt: Market, j: int, when: str, hours: int = 6, floor: float = 0.03) -> Market:
    """Luna-style event: asset ``j`` falls to ``floor`` of its price within ``hours`` hours of
    ``when`` and is then delisted (price frozen, no volume)."""
    t0 = int(mkt.index.searchsorted(pd.Timestamp(when, tz="UTC")))
    n = hours * 60
    f = np.ones(mkt.index.size)
    f[t0 : t0 + n] = np.exp(np.linspace(0, np.log(floor), n))
    f[t0 + n :] = floor
    c = mkt.close.copy(); h = mkt.high.copy(); lo = mkt.low.copy()
    q = mkt.qvol.copy(); v = mkt.vwap.copy()
    for arr in (c, h, lo, v):
        arr[:, j] = arr[:, j] * f
    q[t0 + n :, j] = 0.0
    return Market(mkt.symbols, mkt.index, c, h, lo, q, v)


def stage_stress(ctx: Ctx, when="2025-06-01", seed=None):
    """Delisting stress test: fixed-universe vs periodically re-ranked universe (monthly, weekly)."""
    Af = _universe(ctx, "fixed")
    # crash the least-liquid coin that BOTH the fixed set and the re-ranked set hold at that date
    t_c = int(ctx.bars.time.searchsorted(pd.Timestamp(when, tz="UTC")))
    common = sorted(set(Af[t_c]) & set(rank_universe(ctx.bars, K_UNIV)[t_c]))
    liq = ctx.bars.qvol[t_c - 30 * 48 : t_c].sum(0)
    j = int(min(common, key=lambda i: liq[i]))
    log(f"crash {ctx.mkt.symbols[j]} at {when}")
    mk2 = _crash_market(ctx.mkt, j, when)
    bars2 = mk2.bars(); win2 = Windows(bars2)
    ctx2 = Ctx(mk2, bars2, win2, ctx.sp, Simulator(mk2, bars2), ctx.t_val, ctx.t_test)
    out = {"asset": ctx.mkt.symbols[j], "when": when, "rows": []}
    sel_f = read_json(f"eiie_fixed{K_UNIV}_selection.json")["selected_seed"]
    sel_d = read_json(f"eiie_dyn{K_UNIV}_selection.json")["selected_seed"]
    cases = [("fixed", "fixed", "MS", sel_f), ("dyn-monthly", "dyn", "MS", sel_d),
             ("dyn-weekly", "dyn", "W", sel_d)]
    for label, kind, freq, s in cases:
        A = fixed_universe(bars2, K_UNIV, ctx.sp["train_end"]) if kind == "fixed" else \
            rank_universe(bars2, K_UNIV, freq=freq)
        model = build_eiie(EIIEConfig(n_assets=K_UNIV))
        model.load_state_dict(torch.load(OUT / "models" / f"eiie_{kind}{K_UNIV}_s{s}.pt"))
        W = rollout(model, win2, A, *ctx.t_test)
        for tag, c_ in (("crash", ctx2), ("no-crash", ctx)):
            if tag == "no-crash":
                Wn = rollout(model, ctx.win, A if kind == "fixed" else
                             (rank_universe(ctx.bars, K_UNIV, freq=freq)), *ctx.t_test)
                res = run_sim(ctx, Wn, mode="instant")
            else:
                res = run_sim(ctx2, W, mode="instant")
            out["rows"].append({"universe": label, "scenario": tag, **perf(res)})
            log(f"{label:12s} {tag:9s} fapv {out['rows'][-1]['fapv']:.4f}")
    write_json("stress_delisting.json", out)


# ---------------------------------------------------------------- common table + enhancement 5
MAIN = ["eiie", "sac", "ls_free_1.0", "ls_free_1.5", "ls_free_3.0", "ls_neutral_1.5",
        "eiie_fixed6", "eiie_dyn6", "ucrp_fixed6", "ucrp_dyn6",
        "ubah", "ucrp", "btc_hold", "best_stock_hindsight", "momentum_1d", "reversal_3h",
        "eiie_costblind"]


def stage_main(ctx: Ctx):
    """Every strategy through the *same* engine: paper accounting (instant fill, flat 25 bp),
    $1m book, test period. Also stores down-sampled equity curves for plotting."""
    rows, curves = [], {}
    for name in MAIN:
        if not have_w(name):
            continue
        Wn = load_w(name)
        res = run_sim(ctx, Wn, mode="instant", aum=1e6)
        # time-variation of the chosen weights: ~0 means the policy collapsed to a constant mix
        rows.append({"strategy": name, **perf(res),
                     "weight_time_std": float(Wn[:, 1:].std(axis=0).mean())})
        curves[name] = res.wealth[47::48]
    write_json("main_table.json", rows)
    days = ctx.bars.time[ctx.a : ctx.a + ctx.n_test][47::48]
    np.savez_compressed(OUT / "equity_curves.npz", days=np.array([str(d) for d in days], dtype="U40"),
                        **{k: v for k, v in curves.items()})
    for r in rows:
        log(f"{r['strategy']:22s} fAPV {r['fapv']:.4f}  Sharpe {r['sharpe']:+.2f}  "
            f"MDD {r['max_drawdown_pct']:.1f}%  turn/bar {r['turnover_per_bar']:.4f}")


def stage_attr(ctx: Ctx):
    bk = liquidity_buckets(ctx.bars)
    # liquidity composition of the terciles over the test period
    comp = {}
    for b, nm in enumerate(["low-liquidity", "mid-liquidity", "high-liquidity"]):
        frac = (bk[ctx.a : ctx.a + ctx.n_test] == b).mean(axis=0)
        comp[nm] = {s: round(float(f), 3) for s, f in zip(ctx.mkt.symbols, frac) if f > 0}
    out = {"composition": comp, "tables": {}}
    for name in MAIN:
        if not have_w(name):
            continue
        res = run_sim(ctx, load_w(name), mode="instant", aum=1e6)
        out["tables"][name] = attribute(res, ctx.bars, ctx.a, bk).round(4).to_dict("records")
    write_json("attribution.json", out)
    log("attribution written")


def stage_plots(ctx: Ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "equity_curves.npz")
    days = pd.to_datetime(z["days"])
    groups = {
        "base_vs_benchmarks": ["eiie", "ubah", "ucrp", "btc_hold", "momentum_1d", "reversal_3h"],
        "sac": ["eiie", "sac", "ucrp"],
        "long_short": ["eiie", "ls_free_1.0", "ls_free_1.5", "ls_free_3.0", "ls_neutral_1.5"],
        "universe": ["eiie", "eiie_fixed6", "eiie_dyn6", "ucrp_fixed6", "ucrp_dyn6"],
    }
    for g, names in groups.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for n in names:
            if n in z.files:
                ax.plot(days, z[n], label=n, lw=1.4 if n != "eiie" else 2.4)
        ax.set_ylabel("wealth (start = 1)"); ax.set_title(f"Test period equity: {g}")
        ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(fig_dir / f"equity_{g}.png", dpi=130); plt.close(fig)
    ex = read_json("exec_table.json")
    df = pd.DataFrame(ex)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, aum in zip(axes, (1e6, 1e7)):
        d = df[df.aum == aum].pivot(index="strategy", columns="mode", values="fapv")
        d = d[["instant", "twap", "vwap", "is"]]
        d.plot.bar(ax=ax, width=0.8); ax.set_title(f"final wealth, ${aum/1e6:.0f}M book")
        ax.axhline(1, color="k", lw=0.6); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(fig_dir / "execution_fapv.png", dpi=130); plt.close(fig)
    att = read_json("attribution.json")["tables"]
    names = [n for n in ("eiie", "sac", "ls_free_1.5", "ls_neutral_1.5", "eiie_dyn6", "ucrp") if n in att]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [r["ann_contribution_pct"] for r in att[n]]
        ax.bar(np.arange(3) + i * w, vals, w, label=n)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels(["low-liq", "mid-liq", "high-liq"])
    ax.set_ylabel("annualised net contribution (% of equity)"); ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7); ax.set_title("P&L by liquidity tercile")
    fig.tight_layout(); fig.savefig(fig_dir / "attribution_terciles.png", dpi=130); plt.close(fig)
    log("figures written")
