"""Online Stochastic Batch Learning trainer for the EIIE policy (paper §5.3).

Batch loss follows the original implementation (PGPortfolio ``nnagent.py``):

* the network input ``w_prev`` for every sample is read from the PVM at slot
  ``idx - 1``;
* inside a batch, consecutive samples are chained through the *drifted* network
  output ``future_w`` for the transaction-cost term;
* the cost term uses the first-order turnover approximation
  ``mu ~= 1 - c * sum|future_w[j-1] - w[j]|`` (differentiable). The first sample
  of each batch pays no cost (``mu_0 = 1``);
* loss ``= - mean_j log( mu_j * (w[j] . y_next[j]) )``  (negative of paper eq. 22).

Backtest accounting (evaluation) instead uses the **exact** eq. 14 remainder
factor -- see ``evaluation/backtest.py``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from loguru import logger

from ..models.eiie import EIIEConfig, build_eiie
from .osbl import geometric_batch_start
from .pvm import PVM


@dataclass
class TrainConfig:
    batch_size: int = 50
    steps: int = 80_000
    learning_rate: float = 3e-5
    weight_decay: float = 5e-9
    beta: float = 5e-5
    commission: float = 0.0025
    pvm_init: str = "uniform"
    cost_in_loss: str = "approx"  # 'approx' (REPO) | 'none'
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1000
    grad_clip: float = 0.0


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _batch_loss(
    model: torch.nn.Module,
    X_b: torch.Tensor,
    y_b: torch.Tensor,
    w_prev_b: torch.Tensor,
    commission: float,
    cost_in_loss: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = model(X_b, w_prev_b)  # (B, m+1)
    gross = (w * y_b).sum(dim=1)  # (B,)
    future_w = (y_b * w) / (y_b * w).sum(dim=1, keepdim=True)
    if cost_in_loss == "approx":
        turn = (future_w[:-1] - w[1:]).abs().sum(dim=1)  # (B-1,)
        mu = torch.cat([torch.ones(1, device=w.device), 1.0 - commission * turn])
    elif cost_in_loss == "none":
        mu = torch.ones_like(gross)
    else:
        raise ValueError(cost_in_loss)
    pv = gross * mu
    loss = -torch.log(pv).mean()
    return loss, w.detach()


def train_eiie(
    model_cfg: EIIEConfig,
    cfg: TrainConfig,
    X_train: np.ndarray,
    y_next_train: np.ndarray,
) -> dict:
    """Train from scratch on the training slice. Returns model + PVM snapshot + history."""
    set_determinism(cfg.seed)
    dev = torch.device(cfg.device)
    model = build_eiie(model_cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    N = X_train.shape[0]
    if cfg.batch_size >= N:
        raise ValueError(f"training slice too short: N={N} <= batch_size={cfg.batch_size}")
    pvm = PVM(N, model_cfg.n_assets, init=cfg.pvm_init)
    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y_next_train, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(cfg.seed)

    history = []
    logger.info(
        "train_eiie: N={} steps={} batch={} lr={}", N, cfg.steps, cfg.batch_size, cfg.learning_rate
    )
    for step in range(cfg.steps):
        t_b = geometric_batch_start(N, cfg.batch_size, cfg.beta, rng)
        idx = np.arange(t_b, t_b + cfg.batch_size)
        w_prev = torch.as_tensor(pvm.read_batch(idx - 1), dtype=torch.float32, device=dev)

        opt.zero_grad(set_to_none=True)
        loss, w_new = _batch_loss(
            model, X_t[idx], y_t[idx], w_prev, cfg.commission, cfg.cost_in_loss
        )
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        pvm.write_batch(idx, w_new.cpu().numpy())

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append((step, float(loss.detach())))
            logger.debug("step {} loss {:.6f}", step, float(loss.detach()))

    return {
        "model": model,
        "pvm": pvm.snapshot(),
        "history": history,
        "config": {"model": asdict(model_cfg), "train": asdict(cfg)},
    }
