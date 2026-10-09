"""Baseline portfolio strategies.

Every function takes the risky-asset price-relative matrix

    x[t, i] = close[t, i] / close[t-1, i]          shape (T, m)

and returns an action matrix ``B`` of shape ``(T, m + 1)`` (cash at index 0) where
``B[t]`` is the portfolio chosen at decision time ``t`` using information in
``x[: t + 1]`` only, held over the period whose move is ``y_next[t] = x[t + 1]``.

Classification (see docs/methodology.md):
* cash / UBAH / UCRP / per-asset BAH   -- exact, auditable controls
* best-stock                           -- hindsight, NON-TRADABLE, flagged
* follow-the-winner: EG / ONS / UP     -- reimplementations from the published papers
* follow-the-loser:  PAMR / WMAMR / OLMAR / RMR / CWMR / Anticor -- ditto
* pattern-matching:  CORN              -- ditto (single-rho, history-capped)
* B_K / B_NN (kernel / nearest-neighbour pattern matching), M0, T0 -- NOT implemented
  (documented): very close to CORN, O(T^2), marginal added value here.

All reimplementations follow the formulations in Li & Hoi, "Online Portfolio
Selection: A Survey" (2014) and the original papers; they are *not* validated
against the authors' code and their absolute levels are indicative only.
"""

from __future__ import annotations

import numpy as np

NON_TRADABLE = {"best_stock"}
IMPLEMENTED = [
    "cash",
    "ubah",
    "ucrp",
    "ucrp_risky",
    "best_stock",
    "eg",
    "ons",
    "up",
    "pamr",
    "wmamr",
    "olmar",
    "rmr",
    "cwmr",
    "anticor",
    "corn",
]
NOT_IMPLEMENTED = ["bk", "bnn", "m0", "t0"]


def _with_cash(B_risky: np.ndarray, cash: np.ndarray | float = 0.0) -> np.ndarray:
    B_risky = np.asarray(B_risky, dtype=np.float64)
    T, m = B_risky.shape
    W = np.zeros((T, m + 1))
    W[:, 1:] = B_risky
    W[:, 0] = cash
    # renormalise defensively
    W /= W.sum(axis=1, keepdims=True)
    return W


def _simplex_project(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex."""
    v = np.asarray(v, dtype=np.float64)
    if v.sum() == 1.0 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# auditable controls
# --------------------------------------------------------------------------- #
def cash(x: np.ndarray) -> np.ndarray:
    T, m = x.shape
    W = np.zeros((T, m + 1))
    W[:, 0] = 1.0
    return W


def ucrp(x: np.ndarray) -> np.ndarray:
    """Constant rebalanced to uniform over cash + all risky assets (m+1)."""
    T, m = x.shape
    return np.full((T, m + 1), 1.0 / (m + 1))


def ucrp_risky(x: np.ndarray) -> np.ndarray:
    """Constant rebalanced to uniform over the m risky assets, no cash."""
    T, m = x.shape
    return _with_cash(np.full((T, m), 1.0 / m))


def ubah(x: np.ndarray) -> np.ndarray:
    """Uniform buy-and-hold: start 1/m in each risky asset, never rebalance."""
    T, m = x.shape
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b / b.sum()
        b = b * x[t] if t + 1 < T else b  # drift by the move just realised
        # note: B[t] is the pre-move weight at decision t; drift uses x[t] which is
        # the move into t. Buy-and-hold never trades, so this equals the drift path.
    # rebuild cleanly as the drift of the initial uniform holding
    b = np.full(m, 1.0 / m)
    for t in range(T):
        B[t] = b / b.sum()
        b = b * x[t] if t + 1 < T else b
    return _with_cash(B)


def per_asset_bah(x: np.ndarray) -> dict[int, np.ndarray]:
    """One buy-and-hold action matrix per risky asset (index -> W)."""
    T, m = x.shape
    out = {}
    for i in range(m):
        B = np.zeros((T, m))
        B[:, i] = 1.0
        out[i] = _with_cash(B)
    return out


def best_stock(x: np.ndarray) -> np.ndarray:
    """Hold, for the whole path, the single asset with the best realised return.
    HINDSIGHT -- not tradable; reported only as an upper reference."""
    T, m = x.shape
    final = np.prod(x, axis=0)
    j = int(np.argmax(final))
    B = np.zeros((T, m))
    B[:, j] = 1.0
    return _with_cash(B)


# --------------------------------------------------------------------------- #
# online portfolio selection (reimplementations)
# --------------------------------------------------------------------------- #
def eg(x: np.ndarray, eta: float = 0.05) -> np.ndarray:
    """Exponential Gradient (Helmbold, Schapire, Singer, Warmuth 1998)."""
    T, m = x.shape
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        xt = x[t]
        b = b * np.exp(eta * xt / (b @ xt))
        b /= b.sum()
    return _with_cash(B)


def ons(x: np.ndarray, beta: float = 1.0, delta: float = 1 / 8, eta: float = 0.0) -> np.ndarray:
    """Online Newton Step (Agarwal, Hazan, Kale, Schapire 2006)."""
    T, m = x.shape
    b = np.full(m, 1.0 / m)
    A = np.eye(m)
    p = np.zeros(m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        xt = x[t]
        grad = xt / (b @ xt)
        hess = -np.outer(grad, grad)
        A += -hess
        p += (1.0 + 1.0 / beta) * grad
        b_unproj = delta * np.linalg.solve(A, p)
        b = _simplex_project(b_unproj)
        if eta > 0:
            b = (1 - eta) * b + eta / m
    return _with_cash(B)


def pamr(x: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """Passive Aggressive Mean Reversion, PAMR-0 (Li, Zhao, Hoi, Gopalkrishnan 2012)."""
    T, m = x.shape
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        xt = x[t]
        xbar = xt.mean()
        denom = np.sum((xt - xbar) ** 2)
        loss = max(0.0, b @ xt - eps)
        tau = loss / denom if denom > 1e-12 else 0.0
        b = _simplex_project(b - tau * (xt - xbar))
    return _with_cash(B)


def olmar(x: np.ndarray, window: int = 5, eps: float = 10.0) -> np.ndarray:
    """On-Line Moving Average Reversion, OLMAR-1 (Li & Hoi 2012).
    Works on cumulative price ratios reconstructed from ``x``."""
    T, m = x.shape
    price = np.vstack([np.ones(m), np.cumprod(x, axis=0)])  # (T+1, m), price[0]=1
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        # predicted next relative from the moving average of recent prices
        k = min(window, t + 1)
        pt = price[t + 1]
        ma = np.mean(price[t + 1 - k + 1 : t + 2], axis=0) if k > 1 else pt
        x_pred = ma / pt
        x_mean = x_pred.mean()
        denom = np.sum((x_pred - x_mean) ** 2)
        lam = max(0.0, (eps - b @ x_pred) / denom) if denom > 1e-12 else 0.0
        b = _simplex_project(b + lam * (x_pred - x_mean))
    return _with_cash(B)


def anticor(x: np.ndarray, window: int = 30) -> np.ndarray:
    """BAH(Anticor) single-window variant (Borodin, El-Yaniv, Gogan 2004)."""
    T, m = x.shape
    logx = np.log(x)
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        if t < 2 * window:
            continue
        y1 = logx[t - 2 * window + 1 : t - window + 1]
        y2 = logx[t - window + 1 : t + 1]
        mu1, mu2 = y1.mean(0), y2.mean(0)
        s1, s2 = y1.std(0, ddof=1), y2.std(0, ddof=1)
        cov = ((y1 - mu1).T @ (y2 - mu2)) / (window - 1)
        corr = np.divide(
            cov, np.outer(s1, s2), out=np.zeros_like(cov), where=np.outer(s1, s2) > 1e-12
        )
        claim = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i != j and mu2[i] >= mu2[j] and corr[i, j] > 0:
                    c = corr[i, j]
                    if s1[i] > 1e-12 and s1[j] > 1e-12:
                        c += max(0.0, -corr[i, i]) + max(0.0, -corr[j, j])
                    claim[i, j] += c
        transfer = np.zeros(m)
        for i in range(m):
            tot = claim[i].sum()
            if tot > 1e-12:
                for j in range(m):
                    amt = b[i] * claim[i, j] / tot
                    transfer[i] -= amt
                    transfer[j] += amt
        b = _simplex_project(b + transfer)
    return _with_cash(B)


def up(x: np.ndarray, n_experts: int = 400, seed: int = 0) -> np.ndarray:
    """Universal Portfolios (Cover 1991), Monte-Carlo over Dirichlet CRP experts.

    ``b_{t+1} = sum_e S_e(t) b_e / sum_e S_e(t)`` where ``b_e`` is a fixed CRP and
    ``S_e(t)`` its wealth on ``x[:t+1]``. Exact UP integrates over the simplex; we
    sample ``n_experts`` CRPs once (fixed seed => reproducible)."""
    T, m = x.shape
    rng = np.random.default_rng(seed)
    experts = rng.dirichlet(np.ones(m), size=n_experts)  # (E, m)
    log_wealth = np.zeros(n_experts)
    B = np.empty((T, m))
    for t in range(T):
        w = np.exp(log_wealth - log_wealth.max())
        w /= w.sum()
        B[t] = w @ experts
        log_wealth += np.log(experts @ x[t])
    return _with_cash(B)


def wmamr(x: np.ndarray, window: int = 5, eps: float = 0.5) -> np.ndarray:
    """Weighted Moving Average Mean Reversion (Gao & Zhang 2013): PAMR on the
    moving-average price relative rather than the last one."""
    T, m = x.shape
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        k = min(window, t + 1)
        xt = x[t - k + 1 : t + 1].mean(axis=0)
        xbar = xt.mean()
        denom = np.sum((xt - xbar) ** 2)
        tau = max(0.0, b @ xt - eps) / denom if denom > 1e-12 else 0.0
        b = _simplex_project(b - tau * (xt - xbar))
    return _with_cash(B)


def _l1_median(pts: np.ndarray, iters: int = 30, tol: float = 1e-6) -> np.ndarray:
    """Geometric (L1) median of rows of ``pts`` via Weiszfeld iteration."""
    y = pts.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(pts - y, axis=1)
        nz = d > tol
        if not nz.any():
            return y
        w = 1.0 / d[nz]
        y_new = (w[:, None] * pts[nz]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def rmr(x: np.ndarray, window: int = 5, eps: float = 5.0) -> np.ndarray:
    """Robust Median Reversion (Huang, Zhou, Li, Liu, Zhao 2013): OLMAR with the
    next price predicted by the L1-median of the recent price window."""
    T, m = x.shape
    price = np.vstack([np.ones(m), np.cumprod(x, axis=0)])
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        k = min(window, t + 1)
        pt = price[t + 1]
        med = _l1_median(price[t + 1 - k + 1 : t + 2]) if k > 1 else pt
        x_pred = med / pt
        x_mean = x_pred.mean()
        denom = np.sum((x_pred - x_mean) ** 2)
        lam = max(0.0, (eps - b @ x_pred) / denom) if denom > 1e-12 else 0.0
        b = _simplex_project(b + lam * (x_pred - x_mean))
    return _with_cash(B)


def cwmr(x: np.ndarray, eps: float = 0.5, phi: float = 2.0) -> np.ndarray:
    """Confidence Weighted Mean Reversion, deterministic CWMR-Var
    (Li, Hoi, Zhao, Gopalkrishnan 2011). Keeps a Gaussian ``N(mu, Sigma)`` over
    portfolios; ``phi = Phi^{-1}(confidence)`` (2.0 ~ 0.977)."""
    T, m = x.shape
    mu = np.full(m, 1.0 / m)
    sigma = np.eye(m) / (m**2)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = _simplex_project(mu.copy())
        xt = x[t]
        xbar = (xt @ (sigma @ np.ones(m))) / (np.ones(m) @ sigma @ np.ones(m))
        M = mu @ xt
        V = xt @ sigma @ xt
        if V < 1e-12:
            continue
        a = (M - eps) + phi**2 * V / 2.0
        if a >= 0:
            continue
        disc = a**2 - phi**2 * V * ((M - eps) ** 2 - phi**2 * V**2 / 4.0)
        lam = max(
            0.0,
            (-a - np.sqrt(max(disc, 0.0))) / (phi**2 * V) if phi > 0 else 0.0,
        )
        mu = mu - lam * (sigma @ (xt - xbar))
        sig_inv = np.linalg.inv(sigma) + 2.0 * lam * phi * np.diag(xt**2)
        sigma = np.linalg.inv(sig_inv)
        mu = _simplex_project(mu)
        sigma /= m * np.trace(sigma) + 1e-12
    return _with_cash(B)


def corn(x: np.ndarray, window: int = 5, rho: float = 0.2, max_hist: int = 4000) -> np.ndarray:
    """CORrelation-driven Nonparametric learning, single-rho expert
    (Li, Hoi, Gopalkrishnan 2011). For each step, find past windows whose return
    pattern correlates > ``rho`` with the current window, then hold the CRP that
    maximised wealth over the days that followed those windows.  History is capped
    at ``max_hist`` for tractability on long 30-min series."""
    T, m = x.shape
    logx = np.log(x)
    b = np.full(m, 1.0 / m)
    B = np.empty((T, m))
    for t in range(T):
        B[t] = b
        if t < window + 1:
            continue
        lo = max(window, t - max_hist)
        cur = logx[t - window + 1 : t + 1].ravel()
        cur = cur - cur.mean()
        cn = np.linalg.norm(cur) + 1e-12
        idx = []
        for s in range(lo, t):
            past = logx[s - window + 1 : s + 1].ravel()
            past = past - past.mean()
            if (past @ cur) / (np.linalg.norm(past) * cn + 1e-12) > rho:
                idx.append(s + 1)  # the day AFTER the matched window
        if not idx:
            continue
        follow = x[np.array(idx)]  # (k, m)
        # CRP that maximises product of (b . x) over the followers: log-opt via
        # a few gradient-ascent steps on the simplex.
        w = np.full(m, 1.0 / m)
        for _ in range(15):
            g = (follow / (follow @ w)[:, None]).mean(axis=0)
            w = _simplex_project(w * g)
        b = w
    return _with_cash(B)


ALL = {
    "cash": cash,
    "ubah": ubah,
    "ucrp": ucrp,
    "ucrp_risky": ucrp_risky,
    "best_stock": best_stock,
    "eg": eg,
    "ons": ons,
    "up": up,
    "pamr": pamr,
    "wmamr": wmamr,
    "olmar": olmar,
    "rmr": rmr,
    "cwmr": cwmr,
    "anticor": anticor,
    "corn": corn,
}
