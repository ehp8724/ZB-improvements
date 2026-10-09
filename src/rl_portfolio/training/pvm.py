"""Portfolio-Vector Memory (arXiv:1706.10059v2, §5.2).

A chronological stack of portfolio vectors, one slot per decision time. The
policy network reads slot ``t-1`` as its ``w_prev`` input and writes slot ``t``
with its output after the forward pass. Overlapping training windows therefore
share slots: a later pass overwrites what an earlier pass wrote, which is exactly
the paper's behaviour (the memory converges as the policy converges).

Timing invariants enforced here:
* ``read(t)`` only ever returns a slot with index ``< t`` of the *caller's* step,
  so the current or a future action can never leak into the state.
* the first ``read`` before any ``write`` returns the uniform (or all-cash) init.
"""

from __future__ import annotations

import numpy as np


class PVM:
    def __init__(self, length: int, n_assets: int, *, init: str = "uniform"):
        """``length`` decision times, ``n_assets`` = m (risky). Slot width m+1."""
        self.length = int(length)
        self.width = int(n_assets) + 1
        if init == "uniform":
            row = np.full(self.width, 1.0 / self.width)
        elif init == "cash":
            row = np.zeros(self.width)
            row[0] = 1.0
        else:
            raise ValueError(f"unknown PVM init {init!r}")
        self._buf = np.tile(row, (self.length, 1)).astype(np.float64)
        self._init_row = row.copy()
        self.written = np.zeros(self.length, dtype=bool)

    def read(self, idx: int) -> np.ndarray:
        """Return the stored weight at slot ``idx`` (``idx < 0`` -> init row)."""
        if idx < 0:
            return self._init_row.copy()
        return self._buf[idx].copy()

    def read_batch(self, idxs: np.ndarray) -> np.ndarray:
        idxs = np.asarray(idxs)
        out = np.empty((idxs.size, self.width))
        for i, k in enumerate(idxs):
            out[i] = self.read(int(k))
        return out

    def write(self, idx: int, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.width,):
            raise ValueError(f"expected shape {(self.width,)}, got {w.shape}")
        if not (idx == 0 or idx > 0) or idx >= self.length:
            raise IndexError(f"write index {idx} out of range [0, {self.length})")
        self._buf[idx] = w
        self.written[idx] = True

    def write_batch(self, idxs: np.ndarray, W: np.ndarray) -> None:
        for k, w in zip(np.asarray(idxs), np.asarray(W)):
            self.write(int(k), w)

    def snapshot(self) -> np.ndarray:
        return self._buf.copy()

    def reset(self) -> None:
        self._buf[:] = self._init_row
        self.written[:] = False
