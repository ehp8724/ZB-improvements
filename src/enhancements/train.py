"""Training / rollout for the paper baseline and the enhancement policies.

* ``train_policy``  paper-style OSBL + PVM training of a deterministic policy with the
  differentiable log-growth loss. The same loop trains the long-only EIIE baseline, the
  long/short variants (signed weights, borrow term) and universe-subset models: the
  model sees a (3, K, n) state of the K active assets, while the PVM, transaction costs
  and rewards live in the full universe so a departing coin must really be sold.
* ``train_sac``     Soft Actor-Critic with a Dirichlet policy (see ``sac_*`` below).
* ``rollout``       sequential deployment (previous action is fed back, as in the paper).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rl_portfolio.models.eiie import EIIEConfig, build_eiie
from rl_portfolio.training.osbl import geometric_batch_start
from rl_portfolio.training.pvm import PVM

from .dataset import Windows, gather_prev, scatter_full
from .networks import Critic, DirichletActor


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tA(A: np.ndarray | None, t: np.ndarray | torch.Tensor) -> torch.Tensor | None:
    if A is None:
        return None
    return torch.as_tensor(A[np.asarray(t)], dtype=torch.long)


def cost_growth(w_full, y_full, c, borrow=0.0):
    """Per-sample net growth for consecutive samples (B, m+1) of a batch.

    gross_j = w_j . y_{j}^{next};  the first sample pays no cost (no predecessor).
    Cost is c * sum over *risky* assets of |drifted previous - new| (charged per side, as the
    exact eq.-14 accounting does), plus a borrow charge on short notional.
    """
    gross = (w_full * y_full).sum(1)
    future = w_full * y_full / gross.unsqueeze(1)
    turn = (future[:-1, 1:] - w_full[1:, 1:]).abs().sum(1)
    mu = torch.cat([torch.ones(1), 1.0 - c * turn])
    short = torch.clamp(-w_full[:, 1:], min=0).sum(1)
    return gross * mu - borrow * short, gross


@dataclass
class PolicyCfg:
    steps: int = 30_000
    batch: int = 50
    lr: float = 3e-5
    wd: float = 5e-9
    beta: float = 5e-5
    commission: float = 0.0025
    borrow_per_bar: float = 0.0
    seed: int = 0


def train_policy(model, win: Windows, A: np.ndarray | None, t_lo: int, t_hi: int,
                 cfg: PolicyCfg, log=print) -> dict:
    """Train ``model`` on decision bars ``t_lo..t_hi`` (inclusive). ``A``: (T,K) universe ids."""
    seed_all(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N, m = t_hi - t_lo + 1, win.m
    pvm = PVM(N, m, init="uniform")
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    hist = []
    for step in range(cfg.steps):
        tb = geometric_batch_start(N, cfg.batch, cfg.beta, rng)
        idx = np.arange(tb, tb + cfg.batch)
        t = torch.as_tensor(t_lo + idx)
        At = _tA(A, t_lo + idx)
        wp_full = torch.as_tensor(pvm.read_batch(idx - 1), dtype=torch.float32)
        x = win.state(t, At)
        w = model(x, gather_prev(wp_full, At))
        w_full = scatter_full(w, At, m)
        pv, _ = cost_growth(w_full, win.y_next(t), cfg.commission, cfg.borrow_per_bar)
        loss = -torch.log(pv.clamp_min(1e-4)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pvm.write_batch(idx, w_full.detach().numpy().astype(np.float64))
        if step % 2000 == 0 or step == cfg.steps - 1:
            hist.append((step, float(loss)))
            log(f"  step {step:6d} loss {float(loss):+.6f} ({time.time() - t0:.0f}s)")
    return {"model": model, "history": hist, "pvm": pvm.snapshot()}


@torch.no_grad()
def rollout(policy, win: Windows, A: np.ndarray | None, t_lo: int, t_hi: int,
            deterministic: bool = True) -> np.ndarray:
    """Sequential deployment on decision bars ``t_lo..t_hi``: returns (T, m+1) weights."""
    m = win.m
    T = t_hi - t_lo + 1
    out = np.empty((T, m + 1))
    prev = torch.full((1, m + 1), 1.0 / (m + 1))
    if hasattr(policy, "eval"):
        policy.eval()
    for k in range(T):
        t = torch.as_tensor([t_lo + k])
        At = _tA(A, [t_lo + k])
        x = win.state(t, At)
        wp = gather_prev(prev, At)
        if isinstance(policy, DirichletActor):
            w = policy(x, wp, deterministic=deterministic)[0]
        else:
            w = policy(x, wp)
        prev = scatter_full(w, At, m)
        out[k] = prev[0].numpy()
    return out


def quick_eval(win: Windows, W: np.ndarray, t_lo: int, c: float = 0.0025, borrow: float = 0.0) -> dict:
    """Fast frictional-model evaluation used for validation model selection."""
    T = W.shape[0]
    t = torch.as_tensor(t_lo + np.arange(T))
    w = torch.as_tensor(W, dtype=torch.float32)
    pv, gross = cost_growth(w, win.y_next(t), c, borrow)
    r = (pv - 1.0).numpy()
    return {"fapv": float(np.exp(np.log(np.clip(pv.numpy(), 1e-6, None)).sum())),
            "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(48 * 365)),
            "turnover": float(np.abs(np.diff(W[:, 1:], axis=0)).sum(1).mean())}


# ----------------------------------------------------------------------------- SAC
@dataclass
class SACCfg:
    total_steps: int = 300_000  # environment transitions
    envs: int = 16
    seg: int = 64
    utd_updates: int = 64  # gradient updates after each rollout of envs*seg steps
    batch: int = 128
    gamma: float = 0.9
    tau: float = 0.005
    lr: float = 3e-4
    commission: float = 0.0025
    reward_scale: float = 1e3
    conc_ref: float = 120.0  # reference Dirichlet concentration defining the entropy target
    buffer: int = 300_000
    eval_every: int = 40  # rollouts between validation evaluations
    seed: int = 0


class _Replay:
    def __init__(self, cap: int, m: int):
        self.t = np.zeros(cap, dtype=np.int64)
        self.wp = np.zeros((cap, m + 1), dtype=np.float32)
        self.a = np.zeros((cap, m + 1), dtype=np.float32)
        self.cap, self.n, self.p = cap, 0, 0

    def add(self, t, wp, a):
        for i in range(len(t)):
            self.t[self.p], self.wp[self.p], self.a[self.p] = t[i], wp[i], a[i]
            self.p = (self.p + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, b, rng):
        i = rng.integers(0, self.n, b)
        return self.t[i], torch.as_tensor(self.wp[i]), torch.as_tensor(self.a[i])


def train_sac(win: Windows, A: np.ndarray | None, t_lo: int, t_hi: int, cfg: SACCfg,
              val_range: tuple[int, int] | None = None, log=print) -> dict:
    """Soft Actor-Critic on the portfolio MDP.

    State: price tensor + previous action. Action: simplex weights a ~ Dirichlet.
    Reward: log(a . y_next * (1 - c * |drift(prev) - a|_1)) (net log growth, scaled).
    Transition: next bar, previous action := a. Critics are twin EIIE-style Q networks;
    the entropy temperature is tuned toward the entropy of a reference Dirichlet so the
    actor keeps exploring the allocation simplex instead of collapsing to a constant.
    """
    seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    m = win.m
    K = m if A is None else A.shape[1]
    actor = DirichletActor()
    q1, q2, q1t, q2t = Critic(), Critic(), Critic(), Critic()
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    oa = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    oal = torch.optim.Adam([log_alpha], lr=cfg.lr)
    H_target = actor.target_entropy_ref(K, cfg.conc_ref)
    buf = _Replay(cfg.buffer, m)
    steps_done, rollouts = 0, 0
    best, best_state, hist = -1e9, None, []
    t0 = time.time()

    def reward_and_next(t, wp_full, a_full):
        t_t = torch.as_tensor(t)
        y_prev = win.y[t_t]  # move that drifted the previous action
        drift = wp_full * y_prev / (wp_full * y_prev).sum(1, keepdim=True)
        turn = (drift[:, 1:] - a_full[:, 1:]).abs().sum(1)
        gross = (a_full * win.y_next(t_t)).sum(1)
        return cfg.reward_scale * torch.log((gross * (1.0 - cfg.commission * turn)).clamp_min(1e-4))

    while steps_done < cfg.total_steps:
        # ---- collect: E parallel segments, stochastic policy
        starts = rng.integers(t_lo, t_hi - cfg.seg, cfg.envs)
        wp = torch.full((cfg.envs, m + 1), 1.0 / (m + 1))
        with torch.no_grad():
            for s in range(cfg.seg):
                t = starts + s
                At = _tA(A, t)
                x = win.state(torch.as_tensor(t), At)
                a, _ = actor(x, gather_prev(wp, At))
                a_full = scatter_full(a, At, m)
                buf.add(t, wp.numpy(), a_full.numpy())
                wp = a_full
        steps_done += cfg.envs * cfg.seg
        rollouts += 1
        if buf.n < 4 * cfg.batch:
            continue
        # ---- update
        for _ in range(cfg.utd_updates):
            t, wp_b, a_b = buf.sample(cfg.batch, rng)
            At, Atn = _tA(A, t), _tA(A, np.minimum(t + 1, t_hi))
            r = reward_and_next(t, wp_b, a_b)
            done = torch.as_tensor(t + 1 > t_hi, dtype=torch.float32)
            tn = torch.as_tensor(np.minimum(t + 1, t_hi))
            x, xn = win.state(torch.as_tensor(t), At), win.state(tn, Atn)
            wp_g, a_g, an_prev = gather_prev(wp_b, At), gather_prev(a_b, At), gather_prev(a_b, Atn)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                an, lpn = actor(xn, an_prev)
                qn = torch.min(q1t(xn, an_prev, an), q2t(xn, an_prev, an)) - alpha * lpn
                target = r + cfg.gamma * (1 - done) * qn
            lq = F.mse_loss(q1(x, wp_g, a_g), target) + F.mse_loss(q2(x, wp_g, a_g), target)
            oq.zero_grad(); lq.backward(); oq.step()
            ap, lp = actor(x, wp_g)
            qpi = torch.min(q1(x, wp_g, ap), q2(x, wp_g, ap))
            la = (alpha * lp - qpi).mean()
            oa.zero_grad(); la.backward(); oa.step()
            lal = -(log_alpha * (lp.detach() + H_target)).mean()
            oal.zero_grad(); lal.backward(); oal.step()
            with torch.no_grad():
                for net, tgt in ((q1, q1t), (q2, q2t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        if val_range is not None and rollouts % cfg.eval_every == 0:
            W = rollout(actor, win, A, *val_range)
            ev = quick_eval(win, W, val_range[0], cfg.commission)
            hist.append({"steps": steps_done, **ev, "alpha": float(log_alpha.exp()),
                         "kappa": float(actor.log_kappa.exp()), "q": float(lq)})
            log(f"  sac steps {steps_done:7d} val fAPV {ev['fapv']:.4f} sharpe {ev['sharpe']:+.2f} "
                f"turn {ev['turnover']:.3f} alpha {float(log_alpha.exp()):.3f} "
                f"kappa {float(actor.log_kappa.exp()):.0f} ({time.time() - t0:.0f}s)")
            if ev["sharpe"] > best:
                best, best_state = ev["sharpe"], copy.deepcopy(actor.state_dict())
    if best_state is not None:
        actor.load_state_dict(best_state)
    return {"model": actor, "history": hist, "best_val_sharpe": best}
