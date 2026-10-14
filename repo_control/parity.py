"""Forward and discount factor from consensus call/put prices (put-call parity).

For European options ``C(K) - P(K) = DF * (F - K)``.  A weighted regression of
``C - P`` on ``K`` gives ``slope = -DF`` and ``intercept = DF * F``; the forward
is the zero crossing ``K*`` where ``C = P`` and does not depend on the
discount factor.  The fitted ``DF`` is only a diagnostic (implied funding
spread versus the curve); the controller should keep its own rate curve so
that rates cancel in the FO-vs-Totem repo difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityFit:
    forward: float
    discount: float
    forward_se: float
    residual_rms: float
    n_strikes: int

    def funding_spread(self, curve_df: float, T: float) -> float:
        """Implied funding spread of the quotes versus the curve, ``-ln(DF_fit / DF_curve) / T``."""
        return -np.log(self.discount / curve_df) / T


def forward_from_parity(strikes, calls, puts, weights=None) -> ParityFit:
    """Weighted least-squares parity fit ``C - P = a + c K`` -> ``DF = -c``, ``F = -a / c``."""
    K = np.asarray(strikes, dtype=float)
    y = np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)
    w = np.ones_like(K) if weights is None else np.asarray(weights, dtype=float)
    if K.size < 2:
        raise ValueError("need at least two strikes")
    X = np.column_stack([np.ones_like(K), K])
    WX = X * w[:, None]
    XtWX = X.T @ WX
    coef = np.linalg.solve(XtWX, WX.T @ y)
    a, c = coef
    resid = y - X @ coef
    dof = max(K.size - 2, 1)
    s2 = float((w * resid**2).sum() / dof)
    cov = s2 * np.linalg.inv(XtWX)
    discount = -c
    fwd = -a / c
    # delta method on F = -a / c
    grad = np.array([-1.0 / c, a / c**2])
    se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    return ParityFit(float(fwd), float(discount), se, float(np.sqrt((resid**2).mean())), int(K.size))


def forward_from_parity_fixed_df(strikes, calls, puts, discount: float, weights=None) -> float:
    """Forward with an imposed discount factor: weighted mean of ``K + (C - P) / DF``."""
    K = np.asarray(strikes, dtype=float)
    w = np.ones_like(K) if weights is None else np.asarray(weights, dtype=float)
    est = K + (np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)) / discount
    return float((w * est).sum() / w.sum())
