"""First-order valuation-uncertainty model (sec. 2 of the note).

DeltaPi = <N, dsigma>,  Var(DeltaPi) by Property 1, with the two AVA
extremes: add-up (no netting, eq. 2) and full diversification (eq. 3).

Everything is evaluated on the (M, K) grid in matrix form — ordinary
matrix products and sum reductions only; the (MK, MK) covariance matrix
is never assembled (see :meth:`MarketDataBundle.covariance_of`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .datasource import MarketDataBundle

#: 90% one-sided Gaussian quantile, art. 89 RTS confidence level.
KAPPA_90 = 1.2815515655446004

__all__ = ["UncertaintyModel", "KAPPA_90"]


@dataclass(frozen=True)
class UncertaintyModel:
    """Immutable wrapper precomputing the quadratic-form machinery."""

    bundle: MarketDataBundle
    kappa: float = KAPPA_90

    def variance_of(self, exposure: np.ndarray) -> float:
        """Var(<exposure, dsigma>) — Property 1, matrix sandwich form."""
        return self.bundle.variance_of(exposure)

    def covariance_of(self, expo_a: np.ndarray, expo_b: np.ndarray) -> float:
        return self.bundle.covariance_of(expo_a, expo_b)

    @cached_property
    def var_total(self) -> float:
        """Var(DeltaPi) = Var(<N, dsigma>), Property 1."""
        return self.variance_of(self.bundle.vega)

    @cached_property
    def ava_brut(self) -> float:
        """Add-up AVA, eq. (2): kappa * sum |N_mk| s_mk."""
        return float(self.kappa * np.sum(np.abs(self.bundle.vega) * self.bundle.s))

    @cached_property
    def ava_full(self) -> float:
        """Fully diversified AVA, eq. (3): kappa * sqrt(Var(DeltaPi))."""
        return float(self.kappa * np.sqrt(max(self.var_total, 0.0)))

    def budget(self, alpha: float) -> float:
        """Residual-variance budget B = (1 - alpha) Var(DeltaPi), eq. (6)."""
        return (1.0 - alpha) * self.var_total
