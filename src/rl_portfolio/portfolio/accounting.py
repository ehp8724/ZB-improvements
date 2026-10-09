"""Portfolio wealth accounting for Jiang, Xu & Liang (2017).

All symbols follow the paper (arXiv:1706.10059v2):

* ``w_prev``  : w_{t-1}  -- weights chosen at the end of period t-1 (index 0 = cash).
* ``y``       : y_t      -- price-relative vector between t-1 and t, y_{t,0} = 1.
* ``w_drift`` : w'_t     -- weights after the market move, before rebalancing (Eq. 7).
* ``w``       : w_t      -- target weights chosen at the end of period t (policy output).
* ``mu``      : mu_t     -- transaction-cost remainder factor (Eq. 14).
* wealth recursion : p_t = p_{t-1} * mu_t * (y_t . w_{t-1})     (Eq. 11)
* reward           : r_t = ln( mu_t * (y_t . w_{t-1}) )         (Eq. 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "drifted_weights",
    "transaction_remainder_factor",
    "turnover_approx_remainder",
    "step_wealth",
    "StepResult",
    "simulate_path",
    "backtest_wealth",
]


def drifted_weights(w_prev: np.ndarray, y: np.ndarray) -> np.ndarray:
    """w'_t = (y_t . w_{t-1}) normalised  (paper Eq. 7)."""
    w_prev = np.asarray(w_prev, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = y * w_prev
    denom = num.sum()
    if denom <= 0:
        raise ValueError(f"non-positive portfolio growth factor y.w = {denom!r}")
    return num / denom


def turnover_approx_remainder(w_drift: np.ndarray, w: np.ndarray, c: float) -> float:
    """First-order remainder factor mu ~= 1 - c * sum_i |w'_i - w_i|.

    Used as the iteration seed and as an explicit, documented approximation.
    """
    w_drift = np.asarray(w_drift, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(1.0 - c * np.abs(w_drift - w).sum())


def transaction_remainder_factor(
    w_drift: np.ndarray,
    w: np.ndarray,
    c_s: float,
    c_p: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> float:
    """Solve the implicit equation (paper Eq. 14) for mu_t by fixed-point iteration.

        mu = 1/(1 - c_p w_0) * [ 1 - c_p w'_0
                                 - (c_s + c_p - c_s c_p) * sum_{i>=1} (w'_i - mu w_i)^+ ]

    Index 0 is cash. The map is a contraction on [0, 1] so Banach iteration converges.
    """
    w_drift = np.asarray(w_drift, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w_drift.shape != w.shape:
        raise ValueError("w_drift and w must have the same shape")

    w0_drift = w_drift[0]
    w0 = w[0]
    risky_drift = w_drift[1:]
    risky = w[1:]
    kappa = c_s + c_p - c_s * c_p
    denom = 1.0 - c_p * w0

    mu = max(0.0, min(1.0, turnover_approx_remainder(w_drift, w, 0.5 * (c_s + c_p))))
    for _ in range(max_iter):
        buys = np.maximum(risky_drift - mu * risky, 0.0).sum()
        mu_next = (1.0 - c_p * w0_drift - kappa * buys) / denom
        if abs(mu_next - mu) < tol:
            mu = mu_next
            break
        mu = mu_next
    return float(mu)


@dataclass(frozen=True)
class StepResult:
    wealth: float  # p_t
    log_return: float  # r_t = ln(p_t / p_{t-1})
    mu: float  # mu_t
    cost: float  # transaction cost paid this step in wealth units
    turnover: float  # sum_i |w'_i - w_i|  (fraction of portfolio traded, one side)
    w_drift: np.ndarray  # w'_t
    growth: float  # y_t . w_{t-1}  (gross pre-cost growth factor)


def step_wealth(
    wealth_prev: float,
    w_prev: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    c_s: float,
    c_p: float,
    *,
    use_turnover_approx: bool = False,
) -> StepResult:
    """One period of the wealth recursion, following the paper's order of operations.

    1. start from holdings ``w_prev``           (w_{t-1})
    2. apply the price relative ``y``           (y_t)
    3. compute pre-rebalance weights ``w_drift``(w'_t, Eq. 7)
    4. rebalance towards ``w``                  (w_t, policy output)
    5. apply the transaction-cost factor ``mu`` (mu_t, Eq. 14)
    6. update wealth p_t = p_{t-1} * mu * (y . w_prev)   (Eq. 11)
    """
    w_prev = np.asarray(w_prev, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    growth = float((y * w_prev).sum())
    w_drift = drifted_weights(w_prev, y)
    if use_turnover_approx:
        mu = turnover_approx_remainder(w_drift, w, 0.5 * (c_s + c_p))
    else:
        mu = transaction_remainder_factor(w_drift, w, c_s, c_p)

    wealth_pre_cost = wealth_prev * growth
    wealth = wealth_pre_cost * mu
    cost = wealth_pre_cost * (1.0 - mu)
    turnover = float(np.abs(w_drift - w).sum())
    log_return = float(np.log(mu * growth))
    return StepResult(
        wealth=wealth,
        log_return=log_return,
        mu=mu,
        cost=cost,
        turnover=turnover,
        w_drift=w_drift,
        growth=growth,
    )


def simulate_path(
    weights: np.ndarray,
    price_relatives: np.ndarray,
    c_s: float,
    c_p: float,
    *,
    initial_wealth: float = 1.0,
    w_init: np.ndarray | None = None,
    use_turnover_approx: bool = False,
) -> dict:
    """Roll the wealth recursion over a whole trajectory.

    Arrays are aligned so that at row ``t``:

    * ``price_relatives[t]`` is y_t -- the close-to-close move of the period that
      *ends* at decision time t  (close_t / close_{t-1}), with index 0 == 1 (cash).
    * ``weights[t]``         is w_t -- the action chosen at decision time t from
      information up to and including t, held over the *next* period.

    Step ``t`` therefore drifts the previous action ``weights[t-1]`` by
    ``price_relatives[t]``, rebalances to ``weights[t]`` paying mu_t, and sets
    p_t = p_{t-1} * mu_t * (y_t . w_{t-1})  (paper Eq. 11).  ``w_init`` is w_{-1},
    the holding before the first action (default: all cash).
    """
    weights = np.asarray(weights, dtype=np.float64)
    price_relatives = np.asarray(price_relatives, dtype=np.float64)
    T, n_assets = weights.shape
    if price_relatives.shape != (T, n_assets):
        raise ValueError("weights and price_relatives must have matching shape (T, m+1)")

    if w_init is None:
        w_init = np.zeros(n_assets)
        w_init[0] = 1.0
    w_init = np.asarray(w_init, dtype=np.float64)

    wealth = float(initial_wealth)
    w_prev = w_init
    out = {k: np.empty(T) for k in ("wealth", "log_return", "mu", "cost", "turnover", "growth")}
    out["w_drift"] = np.empty((T, n_assets))

    for t in range(T):
        # action weights[t] were chosen at end of period t; the very first holding
        # is w_init and the market move that acts on w_prev is price_relatives[t].
        res = step_wealth(
            wealth,
            w_prev,
            price_relatives[t],
            weights[t],
            c_s,
            c_p,
            use_turnover_approx=use_turnover_approx,
        )
        wealth = res.wealth
        for k in ("wealth", "log_return", "mu", "cost", "turnover", "growth"):
            out[k][t] = getattr(res, k)
        out["w_drift"][t] = res.w_drift
        w_prev = weights[t]

    out["final_wealth"] = wealth
    out["fapv"] = wealth / initial_wealth
    return out


def backtest_wealth(
    weights: np.ndarray,
    y_next: np.ndarray,
    c_s: float,
    c_p: float,
    *,
    initial_wealth: float = 1.0,
    w_init: np.ndarray | None = None,
    use_turnover_approx: bool = False,
) -> dict:
    """Wealth path for a sequence of *actions* with the evaluation-time alignment.

    ``weights[t]``  is w_t, the action chosen at decision time ``t``.
    ``y_next[t]``   is the close-to-close move realised over the period *after*
                    decision ``t``  (close_{t+1} / close_t), cash column == 1.

    At step ``t`` the previous action ``weights[t-1]`` has drifted under
    ``y_next[t-1]`` to ``w'``; we pay ``mu_t = mu(w', w_t)`` to rebalance, then the
    book grows by ``y_next[t] . w_t``.  ``w_init`` is the holding before ``w_0``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    y_next = np.asarray(y_next, dtype=np.float64)
    T, n_assets = weights.shape
    if y_next.shape != (T, n_assets):
        raise ValueError("weights and y_next must share shape (T, m+1)")
    if w_init is None:
        w_init = np.zeros(n_assets)
        w_init[0] = 1.0
    w_init = np.asarray(w_init, dtype=np.float64)

    cols = ("wealth", "log_return", "simple_return", "mu", "cost", "turnover", "growth")
    out = {k: np.empty(T) for k in cols}
    out["w_drift"] = np.empty((T, n_assets))
    wealth = float(initial_wealth)

    for t in range(T):
        prev_drift = w_init if t == 0 else drifted_weights(weights[t - 1], y_next[t - 1])
        if use_turnover_approx:
            mu = turnover_approx_remainder(prev_drift, weights[t], 0.5 * (c_s + c_p))
        else:
            mu = transaction_remainder_factor(prev_drift, weights[t], c_s, c_p)
        growth = float((y_next[t] * weights[t]).sum())
        wealth_pre = wealth * growth
        wealth = wealth_pre * mu
        out["wealth"][t] = wealth
        out["growth"][t] = growth
        out["mu"][t] = mu
        out["cost"][t] = wealth_pre * (1.0 - mu)
        out["turnover"][t] = float(np.abs(prev_drift - weights[t]).sum())
        out["log_return"][t] = float(np.log(mu * growth))
        out["simple_return"][t] = mu * growth - 1.0
        out["w_drift"][t] = prev_drift

    out["final_wealth"] = wealth
    out["fapv"] = wealth / initial_wealth
    return out
