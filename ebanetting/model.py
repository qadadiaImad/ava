"""First-order valuation-uncertainty model (sec. 2 of the note).

DeltaPi = nu' dsigma,  Var(DeltaPi) = nu' Sigma nu, with the two AVA
extremes: add-up (no netting, eq. 2) and full diversification (eq. 3).
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

    @cached_property
    def nu(self) -> np.ndarray:
        return self.bundle.nu

    @cached_property
    def sigma(self) -> np.ndarray:
        return self.bundle.covariance()

    @cached_property
    def var_total(self) -> float:
        """Var(DeltaPi) = nu' Sigma nu, eq. (1)."""
        return float(self.nu @ self.sigma @ self.nu)

    @cached_property
    def ava_brut(self) -> float:
        """Add-up AVA, eq. (2): kappa * sum |nu_i| s_i."""
        return float(self.kappa * np.sum(np.abs(self.nu) * self.bundle.s_vec))

    @cached_property
    def ava_full(self) -> float:
        """Fully diversified AVA, eq. (3): kappa * sqrt(nu' Sigma nu)."""
        return float(self.kappa * np.sqrt(max(self.var_total, 0.0)))

    def budget(self, alpha: float) -> float:
        """Residual-variance budget B = (1 - alpha) Var(DeltaPi), eq. (7)."""
        return (1.0 - alpha) * self.var_total
