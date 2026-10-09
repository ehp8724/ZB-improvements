"""EIIE policy networks (arXiv:1706.10059v2, §5).

The paper implements the Ensemble of Identical Independent Evaluators with three
interchangeable per-asset evaluators -- a CNN, a basic RNN and an LSTM -- sharing
the same scoring tail (previous-weight feature map -> 1x1 conv -> cash bias ->
softmax). All three are provided here and selected by ``EIIEConfig.kind``.

CNN shape trace (defaults ``n=50``, ``m=11``, ``f1=2``, ``f2=20``):

    X            (B, 3, m, n)
    conv1 1x3    (B, f1, m, n-2)     ReLU
    conv2 1x(n-2)(B, f2, m, 1)       ReLU        "EIIE dense"
    [+ w_prev]   (B, f2+1, m, 1)     previous risky weights as one feature map
    conv3 1x1    (B, 1, m, 1) -> (B, m)          "scoring layer"
    + cash bias  (B, m+1)
    softmax      (B, m+1)                         portfolio weights w_t

RNN / LSTM: each asset's length-``n`` feature sequence is run through **one shared
recurrent cell** (weights identical across assets -- the "identical independent
evaluator"); the last hidden state (``hidden`` units) replaces the conv2 output,
then the identical scoring tail follows.  (PGPortfolio ``EIIE_RNN`` /
``EIIE_LSTM``: hidden 20, LSTM dropout 0.5.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EIIEConfig", "EIIE_CNN", "EIIE_RNN", "EIIE_LSTM", "build_eiie"]


@dataclass
class EIIEConfig:
    n_assets: int = 11  # m, risky only
    window: int = 50  # n
    in_channels: int = 3  # close, high, low  (1 => close-only ablation)
    f1: int = 2  # feature maps after conv1  (REPO cfg: 3)
    f2: int = 20  # feature maps after conv2  (REPO cfg: 10)
    kernel1_width: int = 3  # conv1 time-kernel width   (REPO cfg: 2)
    use_prev_weights: bool = True  # PVM / w_{t-1} input (ablation toggle)
    kind: str = "cnn"  # "cnn" | "rnn" | "lstm"
    hidden: int = 20  # recurrent hidden units (rnn/lstm)
    dropout: float = 0.0  # dropout on the recurrent hidden state (paper LSTM: 0.5)


def _score_tail(
    feats: torch.Tensor,  # (B, F, m, 1)  per-asset feature stack
    w_prev: torch.Tensor,  # (B, m+1)
    conv_score: nn.Conv2d,
    cash_bias: nn.Parameter,
    use_prev_weights: bool,
) -> torch.Tensor:
    """Shared EIIE scoring tail -> portfolio weights on the (m+1)-simplex."""
    B, _, m, _ = feats.shape
    if use_prev_weights:
        wp = w_prev[:, 1:].reshape(B, 1, m, 1)  # drop cash column
        feats = torch.cat([feats, wp], dim=1)
    scores = conv_score(feats).reshape(B, m)  # (B, m)
    logits = torch.cat([cash_bias.expand(B, 1), scores], dim=1)  # (B, m+1)
    return F.softmax(logits, dim=-1)


class EIIE_CNN(nn.Module):
    def __init__(self, cfg: EIIEConfig):
        super().__init__()
        self.cfg = cfg
        w2 = cfg.window - cfg.kernel1_width + 1  # time dim after conv1
        self.conv1 = nn.Conv2d(cfg.in_channels, cfg.f1, kernel_size=(1, cfg.kernel1_width))
        self.conv2 = nn.Conv2d(cfg.f1, cfg.f2, kernel_size=(1, w2))
        score_in = cfg.f2 + (1 if cfg.use_prev_weights else 0)
        self.conv3 = nn.Conv2d(score_in, 1, kernel_size=(1, 1))
        self.cash_bias = nn.Parameter(torch.zeros(1))
        self._w2 = w2

    def forward(self, x: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        """``x``: (B, in_channels, m, n).  ``w_prev``: (B, m+1) incl. cash at 0."""
        cfg = self.cfg
        B = x.shape[0]
        assert x.shape[1:] == (cfg.in_channels, cfg.n_assets, cfg.window), x.shape
        assert w_prev.shape == (B, cfg.n_assets + 1), w_prev.shape

        h = F.relu(self.conv1(x))
        assert h.shape == (B, cfg.f1, cfg.n_assets, self._w2), h.shape
        h = F.relu(self.conv2(h))
        assert h.shape == (B, cfg.f2, cfg.n_assets, 1), h.shape

        w = _score_tail(h, w_prev, self.conv3, self.cash_bias, cfg.use_prev_weights)
        assert w.shape == (B, cfg.n_assets + 1), w.shape
        return w


class _EIIERecurrent(nn.Module):
    """Shared machinery for the RNN and LSTM evaluators."""

    def __init__(self, cfg: EIIEConfig, cell: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.cell = cell  # nn.RNN or nn.LSTM, batch_first, input_size=in_channels
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        score_in = cfg.hidden + (1 if cfg.use_prev_weights else 0)
        self.score = nn.Conv2d(score_in, 1, kernel_size=(1, 1))
        self.cash_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = x.shape[0]
        m, n = cfg.n_assets, cfg.window
        assert x.shape[1:] == (cfg.in_channels, m, n), x.shape
        assert w_prev.shape == (B, m + 1), w_prev.shape

        # (B, C, m, n) -> (B*m, n, C): one sequence per asset, cell weights shared
        seq = x.permute(0, 2, 3, 1).reshape(B * m, n, cfg.in_channels).contiguous()
        out, _ = self.cell(seq)  # (B*m, n, hidden)
        last = self.drop(out[:, -1, :])  # (B*m, hidden)
        feats = last.reshape(B, m, cfg.hidden).permute(0, 2, 1).unsqueeze(-1)  # (B, hidden, m, 1)

        w = _score_tail(feats, w_prev, self.score, self.cash_bias, cfg.use_prev_weights)
        assert w.shape == (B, m + 1), w.shape
        return w


class EIIE_RNN(_EIIERecurrent):
    def __init__(self, cfg: EIIEConfig):
        super().__init__(
            cfg,
            nn.RNN(cfg.in_channels, cfg.hidden, batch_first=True, nonlinearity="tanh"),
        )


class EIIE_LSTM(_EIIERecurrent):
    def __init__(self, cfg: EIIEConfig):
        super().__init__(cfg, nn.LSTM(cfg.in_channels, cfg.hidden, batch_first=True))


_REGISTRY = {"cnn": EIIE_CNN, "rnn": EIIE_RNN, "lstm": EIIE_LSTM}


def build_eiie(cfg: EIIEConfig) -> nn.Module:
    """Instantiate the evaluator named by ``cfg.kind``."""
    try:
        return _REGISTRY[cfg.kind](cfg)
    except KeyError:
        raise ValueError(
            f"unknown EIIE kind {cfg.kind!r}; choose from {sorted(_REGISTRY)}"
        ) from None
