"""Causal policy rollout + exact-cost portfolio backtest.

The rollout is strictly sequential: the action at decision ``t`` is produced from
the state ``X[t]`` (bars ``<= t``) and the **realised** previous action
``W[t-1]`` (``W[-1] = w_init``). No future information and no PVM slot ahead of
``t`` is ever read. Wealth is then accounted with the exact eq.-14 remainder
factor via ``accounting.backtest_wealth``.

Optional online learning (paper §5.3): after emitting ``W[t]`` we may take a few
SGD steps on batches sampled from history ``<= t`` only.
"""

from __future__ import annotations

import numpy as np
import torch

from ..portfolio.accounting import backtest_wealth
from ..training.osbl import geometric_batch_start
from ..training.trainer import TrainConfig, _batch_loss


@torch.no_grad()
def _policy_step(model, x_t: np.ndarray, w_prev: np.ndarray, device) -> np.ndarray:
    X = torch.as_tensor(x_t[None], dtype=torch.float32, device=device)
    wp = torch.as_tensor(w_prev[None], dtype=torch.float32, device=device)
    return model(X, wp).cpu().numpy()[0]


def rollout(
    model,
    X: np.ndarray,
    y_next: np.ndarray,
    *,
    commission: float = 0.0025,
    w_init: np.ndarray | None = None,
    device: str = "cpu",
    online: bool = False,
    online_cfg: TrainConfig | None = None,
    online_steps: int = 0,
    hist_X: np.ndarray | None = None,
    hist_y: np.ndarray | None = None,
) -> dict:
    """Roll the policy over an evaluation slice and account wealth with exact costs.

    ``X``, ``y_next`` are the evaluation samples in chronological order.
    For online learning, ``hist_X``/``hist_y`` are the samples strictly before the
    slice (training + any already-seen validation); each step appends the newly
    revealed sample and trains only on that growing buffer.
    """
    dev = torch.device(device)
    model = model.to(dev)
    m1 = X.shape[2] + 1
    if w_init is None:
        w_init = np.zeros(m1)
        w_init[0] = 1.0

    N = X.shape[0]
    W = np.empty((N, m1))
    w_prev = w_init.copy()

    opt = None
    if online and online_steps > 0:
        online_cfg = online_cfg or TrainConfig()
        opt = torch.optim.Adam(
            model.parameters(), lr=online_cfg.learning_rate, weight_decay=online_cfg.weight_decay
        )
        rng = np.random.default_rng(online_cfg.seed + 1)
        # one contiguous chronological buffer: [history .. | .. test], indexed, never re-stacked
        pre_X = np.concatenate([hist_X, X]) if hist_X is not None else X
        pre_y = np.concatenate([hist_y, y_next]) if hist_y is not None else y_next
        all_X = torch.as_tensor(pre_X, dtype=torch.float32, device=dev)
        all_y = torch.as_tensor(pre_y, dtype=torch.float32, device=dev)
        n_hist = len(pre_X) - N

    for t in range(N):
        model.eval()
        W[t] = _policy_step(model, X[t], w_prev, dev)
        w_prev = W[t]

        if online and online_steps > 0:
            assert online_cfg is not None and opt is not None
            avail = n_hist + t + 1  # samples revealed up to and including step t
            if avail > online_cfg.batch_size + 1:
                model.train()
                for _ in range(online_steps):
                    tb = geometric_batch_start(avail, online_cfg.batch_size, online_cfg.beta, rng)
                    sl = slice(tb, tb + online_cfg.batch_size)
                    wpb = torch.full((online_cfg.batch_size, m1), 1.0 / m1, device=dev)
                    opt.zero_grad(set_to_none=True)
                    loss, _ = _batch_loss(
                        model,
                        all_X[sl],
                        all_y[sl],
                        wpb,
                        online_cfg.commission,
                        online_cfg.cost_in_loss,
                    )
                    loss.backward()
                    opt.step()

    acct = backtest_wealth(W, y_next, commission, commission, w_init=w_init)
    acct["weights"] = W
    return acct


def rollout_weights_only(
    model, X: np.ndarray, w_init: np.ndarray | None = None, device: str = "cpu"
) -> np.ndarray:
    """Just the action matrix W (N, m+1) from a causal rollout (no accounting)."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    m1 = X.shape[2] + 1
    if w_init is None:
        w_init = np.zeros(m1)
        w_init[0] = 1.0
    W = np.empty((X.shape[0], m1))
    w_prev = w_init.copy()
    for t in range(X.shape[0]):
        W[t] = _policy_step(model, X[t], w_prev, dev)
        w_prev = W[t]
    return W
