"""Networks for the enhancements. All reuse the paper's EIIE per-asset CNN trunk."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trunk(nn.Module):
    """Paper EIIE evaluator: 1x3 conv -> 1x(n-2) conv, identical across assets."""

    def __init__(self, n: int = 50, in_ch: int = 3, f1: int = 2, f2: int = 20, k1: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, f1, (1, k1))
        self.conv2 = nn.Conv2d(f1, f2, (1, n - k1 + 1))
        self.f2 = f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,K,n) -> (B,f2,K,1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class LongShortEIIE(nn.Module):
    """EIIE with signed weights under a gross-leverage cap (enhancement 3).

    score_i -> a_i = tanh(score_i) in (-1, 1).  ``neutral`` removes the cross-sectional
    mean so net exposure is zero (pure relative value).  Weights are scaled *down only*
    so that gross = sum|w| <= ``max_gross``; cash is the residual 1 - sum(w).
    """

    def __init__(self, n: int = 50, max_gross: float = 1.5, neutral: bool = False,
                 f1: int = 2, f2: int = 20):
        super().__init__()
        self.trunk = Trunk(n, f1=f1, f2=f2)
        self.score = nn.Conv2d(f2 + 1, 1, (1, 1))
        self.max_gross, self.neutral = max_gross, neutral

    def forward(self, x: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        B, _, K, _ = x.shape
        h = self.trunk(x)
        h = torch.cat([h, w_prev[:, 1:].reshape(B, 1, K, 1)], dim=1)
        a = torch.tanh(self.score(h).reshape(B, K))
        if self.neutral:
            a = a - a.mean(dim=1, keepdim=True)
        gross = a.abs().sum(dim=1, keepdim=True)
        w = a * torch.clamp(self.max_gross / (gross + 1e-8), max=1.0)
        return torch.cat([1.0 - w.sum(dim=1, keepdim=True), w], dim=1)


class DirichletActor(nn.Module):
    """SAC actor: Dirichlet(kappa * softmax(scores)) over the K+1 simplex (cash first).

    The mean allocation is an EIIE-style softmax of per-asset scores; ``kappa`` (total
    concentration, learned) sets how exploratory the policy is.
    """

    def __init__(self, n: int = 50, f1: int = 2, f2: int = 20, kappa0: float = 60.0):
        super().__init__()
        self.trunk = Trunk(n, f1=f1, f2=f2)
        self.score = nn.Conv2d(f2 + 1, 1, (1, 1))
        self.cash_bias = nn.Parameter(torch.zeros(1))
        self.log_kappa = nn.Parameter(torch.tensor(float(kappa0)).log())

    def dist_params(self, x, w_prev):
        B, _, K, _ = x.shape
        h = torch.cat([self.trunk(x), w_prev[:, 1:].reshape(B, 1, K, 1)], dim=1)
        logits = torch.cat([self.cash_bias.expand(B, 1), self.score(h).reshape(B, K)], dim=1)
        mean = F.softmax(logits, dim=-1)
        kappa = self.log_kappa.exp().clamp(K + 1.0, 2000.0)
        return (mean * kappa).clamp_min(1e-3), mean

    def forward(self, x, w_prev, deterministic: bool = False):
        alpha, mean = self.dist_params(x, w_prev)
        if deterministic:
            return mean, None
        d = torch.distributions.Dirichlet(alpha)
        a = d.rsample().clamp_min(1e-6)
        a = a / a.sum(-1, keepdim=True)
        return a, d.log_prob(a)

    def target_entropy_ref(self, K: int, conc: float) -> float:
        d = torch.distributions.Dirichlet(torch.full((K + 1,), conc / (K + 1)))
        return float(d.entropy())


class Critic(nn.Module):
    """Q(s, a): per-asset EIIE features + (prev weight, action weight), pooled, MLP."""

    def __init__(self, n: int = 50, f1: int = 2, f2: int = 20, hid: int = 32):
        super().__init__()
        self.trunk = Trunk(n, f1=f1, f2=f2)
        self.per_asset = nn.Conv2d(f2 + 2, hid, (1, 1))
        self.head = nn.Sequential(nn.Linear(2 * hid + 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, w_prev, a):
        B, _, K, _ = x.shape
        f = self.trunk(x)
        z = torch.cat([f, w_prev[:, 1:].reshape(B, 1, K, 1), a[:, 1:].reshape(B, 1, K, 1)], dim=1)
        z = F.relu(self.per_asset(z)).reshape(B, -1, K)
        pooled = torch.cat([z.mean(-1), z.max(-1).values, w_prev[:, :1], a[:, :1]], dim=1)
        return self.head(pooled).squeeze(-1)
