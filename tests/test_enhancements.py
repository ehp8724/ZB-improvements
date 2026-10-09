import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import torch

from enhancements.attribution import attribute, liquidity_buckets
from enhancements.data import BAR, Market
from enhancements.dataset import Windows
from enhancements.execution import ExecConfig, Simulator, schedule
from enhancements.networks import DirichletActor, LongShortEIIE
from enhancements.universe import fixed_universe, rank_universe
from rl_portfolio.features.price_tensor import price_tensor_at

SYMS = ["BTC", "ETH", "ADA"]


def make_market(nbars=400, seed=0):
    rng = np.random.default_rng(seed)
    M = nbars * BAR
    idx = pd.date_range("2024-01-01", periods=M, freq="1min", tz="UTC")
    r = rng.normal(0, 5e-4, (M, 3))
    close = 100 * np.exp(np.cumsum(r, 0))
    high, low = close * 1.0005, close * 0.9995
    q = rng.uniform(5e4, 2e5, (M, 3)) * np.array([5.0, 2.0, 0.5])
    return Market(SYMS, idx, close, high, low, q, close.copy())


@pytest.fixture(scope="module")
def sim():
    mk = make_market()
    return Simulator(mk, mk.bars())


@pytest.mark.parametrize("mode", ["instant", "twap", "vwap", "is"])
def test_pnl_identity_and_borrow(sim, mode):
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.4, (200, 3))  # signed, leveraged
    res = sim.run(W, 10, ExecConfig(mode=mode, aum=2e5))
    d_equity = res.equity[-1] - res.equity[0]
    assert res.asset_pnl.sum() - res.borrow.sum() == pytest.approx(d_equity, abs=1e-6)
    # gross return + costs reconcile to net return
    costs = (res.fee + res.slip + res.borrow)
    assert np.allclose(res.gross_ret * res.equity[:-1] - costs, res.ret * res.equity[:-1], atol=1e-6)


def test_instant_constant_weights_only_pays_initial_fee(sim):
    W = np.tile([0.5, 0.3, 0.2], (50, 1))
    res = sim.run(W, 5, ExecConfig(mode="instant", fee=0.0025, aum=1e6))
    assert res.fee[0] == pytest.approx(0.0025 * 1e6, rel=1e-9)  # entering the book
    assert res.fee.sum() > res.fee[0]  # later bars rebalance back to constant mix after drift
    assert res.slip.sum() == 0


def test_schedules():
    for mode in ["twap", "is"]:
        s = schedule(mode, 30, 3.0, None)
        assert s.sum() == pytest.approx(1.0)
    assert np.allclose(schedule("twap", 30, 0, None), 1 / 30)
    s = schedule("is", 30, 4.0, None)
    assert s[0] > s[-1] and np.all(np.diff(s) < 0)  # front-loaded
    prof = np.random.default_rng(0).uniform(1, 2, (30, 3))
    v = schedule("vwap", 30, 0, prof)
    assert np.allclose(v.sum(0), 1.0)


def test_impact_grows_with_size(sim):
    W = np.tile([0.4, 0.3, 0.3], (60, 1)); W[1::2] = [0.2, 0.2, 0.6]
    small = sim.run(W, 10, ExecConfig(mode="twap", aum=1e5)).slip.sum() / 1e5
    big = sim.run(W, 10, ExecConfig(mode="twap", aum=1e7)).slip.sum() / 1e7
    assert big > 3 * small


def test_windows_match_paper_tensor():
    mk = make_market(120)
    b = mk.bars()
    win = Windows(b, n=50)
    t = 70
    got = win.state(torch.tensor([t]))[0].numpy()
    want = price_tensor_at(b.close, b.high, b.low, t, 50)
    assert np.allclose(got, want, rtol=1e-5)
    sub = win.state(torch.tensor([t]), torch.tensor([[2, 0]]))[0].numpy()
    assert np.allclose(sub, want[:, [2, 0], :], rtol=1e-5)


def test_long_short_constraints():
    net = LongShortEIIE(max_gross=1.5, neutral=False)
    net2 = LongShortEIIE(max_gross=1.5, neutral=True)
    x = torch.rand(8, 3, 5, 50) + 0.5
    wp = torch.full((8, 6), 1 / 6)
    for n in (net, net2):
        w = n(x, wp)
        assert torch.allclose(w.sum(1), torch.ones(8), atol=1e-5)  # cash is the residual
        assert (w[:, 1:].abs().sum(1) <= 1.5 + 1e-5).all()
    assert net2(x, wp)[:, 1:].sum(1).abs().max() < 1e-5  # dollar neutral


def test_dirichlet_actor_on_simplex():
    act = DirichletActor()
    x = torch.rand(6, 3, 4, 50) + 0.5
    wp = torch.full((6, 5), 0.2)
    a, lp = act(x, wp)
    assert torch.allclose(a.sum(1), torch.ones(6), atol=1e-5) and (a >= 0).all()
    assert torch.isfinite(lp).all()
    a0, _ = act(x, wp, deterministic=True)
    assert torch.allclose(a0.sum(1), torch.ones(6), atol=1e-5)


def test_universe_has_no_lookahead():
    mk = make_market(48 * 70)
    b = mk.bars()
    A1 = rank_universe(b, 2)
    b2 = mk.bars()
    cut = 48 * 40
    b2.qvol[cut:] = b2.qvol[cut:][:, ::-1]  # scramble the future
    A2 = rank_universe(b2, 2)
    assert np.array_equal(A1[:cut], A2[:cut])
    F = fixed_universe(b, 2, at_bar=100)
    assert (F == F[0]).all()


def test_attribution_sums_to_total(sim):
    b = sim.bars
    rng = np.random.default_rng(2)
    W = rng.dirichlet(np.ones(3), 150)
    res = sim.run(W, 60, ExecConfig(mode="instant", aum=1e6))
    tab = attribute(res, b, 60, liquidity_buckets(b))
    assert tab["net_pnl_usd"].sum() == pytest.approx(res.asset_pnl.sum(), rel=1e-9)
