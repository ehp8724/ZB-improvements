"""Price-tensor state and price-relative construction (arXiv:1706.10059v2, §3.2).

Conventions
-----------
* ``close/high/low`` are float arrays of shape ``(T, m)`` -- ``m`` risky assets,
  rows ordered by time, quoted in the numeraire (BTC in the paper, USDT here).
* Asset order is fixed and identical everywhere: data, tensors, PVM, outputs.
* The state at decision time ``t`` uses only bars with index ``<= t`` (no look-ahead).
"""

from __future__ import annotations

import numpy as np

FEATURES = ("close", "high", "low")


def price_relatives(close: np.ndarray) -> np.ndarray:
    """y_t = v_t ⊘ v_{t-1} for the risky assets, with a cash column of ones.

    Returns shape ``(T, m + 1)`` (index 0 = cash). Row 0 is all ones (no prior bar).
    """
    close = np.asarray(close, dtype=np.float64)
    T, m = close.shape
    y = np.ones((T, m + 1), dtype=np.float64)
    y[1:, 1:] = close[1:] / close[:-1]
    return y


def price_tensor_at(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    t: int,
    n: int,
) -> np.ndarray:
    """State ``X_t`` of shape ``(3, m, n)`` = (feature, asset, time).

    Window is bar indices ``t-n+1 .. t`` inclusive. Every channel is divided by
    the **latest close** ``close[t]`` of each asset, so ``X_t[0, :, -1] == 1``
    (paper eq. 18). No cash row here -- cash enters only via ``w_prev`` and softmax.
    """
    if t - n + 1 < 0:
        raise ValueError(f"need t-n+1 >= 0, got t={t}, n={n}")
    sl = slice(t - n + 1, t + 1)
    ref = close[t]  # shape (m,), the decision-time close
    stack = np.stack(
        [close[sl].T / ref[:, None], high[sl].T / ref[:, None], low[sl].T / ref[:, None]],
        axis=0,
    )
    return stack.astype(np.float64)


def build_windows(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised dataset of overlapping states and the following price relative.

    Returns
    -------
    X : ``(N, 3, m, n)``   state tensors for decision times ``t = n-1 .. T-2``
    y_next : ``(N, m+1)``  price relative realised over the step *after* the state,
             i.e. ``close[t+1] / close[t]`` with a leading cash 1.
    t_index : ``(N,)``     the bar index ``t`` each sample was taken at.
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    T, m = close.shape
    if n + 1 > T:
        raise ValueError(f"need at least n+1={n + 1} bars, got {T}")

    t_index = np.arange(n - 1, T - 1)
    N = t_index.size
    X = np.empty((N, 3, m, n), dtype=np.float64)
    for i, t in enumerate(t_index):
        X[i] = price_tensor_at(close, high, low, t, n)

    y = price_relatives(close)  # (T, m+1)
    y_next = y[t_index + 1]  # move over the period following each state
    return X, y_next, t_index
