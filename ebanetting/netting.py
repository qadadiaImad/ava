"""Netting as an aggregation operator and the EBA variance test
(sections 4 and 5 of the note).

A netting scheme is a partition P of the grid cells, encoded as an (M, K)
integer label matrix. Each set r carries a representative shock
d~sigma_r = <W_r, dsigma> with W_r a row-stochastic (M, K) weight grid,
netted exposure m_r = sum of N over the set (Def. 4), so that

    proxy        = sum_r m_r <W_r, dsigma>
    residual     = N - sum_r m_r W_r            (an (M, K) matrix)
    TE^2(P)      = Var(<residual, dsigma>)                       (6)
    AVA(P)       = kappa * sum_r |m_r| ~s_r            (Def. 4)
    variance ok  <=> TE^2 <= (1 - alpha) Var(DeltaPi)            (6)
    floor ok     <=> AVA(P) >= kappa sqrt(Var(DeltaPi))          (7)

All variances are computed in matrix form (Property 1) — ordinary
matrix products and reductions, no Kronecker / tensor product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .model import UncertaintyModel

Weighting = Literal["pivot", "vega", "equal"]

__all__ = ["NettingScheme", "SetStat", "SchemeEvaluation", "two_bucket"]


def _normalise_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel sets as 0..K-1 in order of first appearance (row-major)."""
    labels = np.asarray(labels, dtype=int)
    flat = labels.flatten()
    _, first = np.unique(flat, return_index=True)
    order = flat[np.sort(first)]
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.get)(labels)


@dataclass(frozen=True)
class NettingScheme:
    """Partition of the (tenor, strike) grid with representative-shock
    weights."""

    labels: np.ndarray          # (M, K) int, sets numbered 0..K-1
    weighting: Weighting = "pivot"

    @classmethod
    def from_labels(cls, labels: np.ndarray, weighting: Weighting = "pivot") -> "NettingScheme":
        return cls(labels=_normalise_labels(labels), weighting=weighting)

    @classmethod
    def singletons(cls, M: int, K: int, weighting: Weighting = "pivot") -> "NettingScheme":
        return cls(labels=np.arange(M * K).reshape(M, K), weighting=weighting)

    @property
    def n_sets(self) -> int:
        return int(self.labels.max()) + 1

    def members(self, k: int) -> np.ndarray:
        """Flat (row-major) bucket indices of set k."""
        return np.flatnonzero(self.labels.flatten() == k)

    # ------------------------------------------------------------------ #
    def weight_grids(self, model: UncertaintyModel) -> np.ndarray:
        """(n_sets, M, K) representative-shock weights — one row-stochastic
        grid per set (Def. 4 of the note)."""
        vega, s = model.bundle.vega, model.bundle.s
        out = np.zeros((self.n_sets, *self.labels.shape))
        for k in range(self.n_sets):
            mask = self.labels == k
            if self.weighting == "equal":
                out[k][mask] = 1.0 / mask.sum()
            elif self.weighting == "vega":
                w = np.where(mask, np.abs(vega), 0.0)
                tot = w.sum()
                out[k] = w / tot if tot > 0 else mask / mask.sum()
            else:  # pivot: risk-dominant cell |N_mk| s_mk of the set
                score = np.where(mask, np.abs(vega) * s, -np.inf)
                out[k][np.unravel_index(np.argmax(score), score.shape)] = 1.0
        return out

    def evaluate(self, model: UncertaintyModel, alpha: float) -> "SchemeEvaluation":
        """Run the variance test (6) and the conservatism floor (7)."""
        vega, s = model.bundle.vega, model.bundle.s
        grids = self.weight_grids(model)
        m = np.array([float(vega[self.labels == k].sum()) for k in range(self.n_sets)])
        # ~s_r^2 = Var(<W_r, dsigma>) — Property 1 in matrix form
        s_tilde = np.sqrt(np.clip([model.variance_of(g) for g in grids], 0.0, None))
        proxy = np.zeros_like(vega, dtype=float)
        for m_k, grid in zip(m, grids):
            proxy += m_k * grid
        residual = vega - proxy                      # N - sum_r m_r W_r
        te2 = max(model.variance_of(residual), 0.0)
        var_total = model.var_total
        r2 = 1.0 - te2 / var_total if var_total > 0 else 1.0
        ava = float(model.kappa * np.sum(np.abs(m) * s_tilde))
        budget = model.budget(alpha)

        stats = []
        for k in range(self.n_sets):
            mask = self.labels == k
            stats.append(
                SetStat(
                    set_id=k,
                    size=int(mask.sum()),
                    net_vega=float(m[k]),
                    gross_vega=float(np.abs(vega[mask]).sum()),
                    s_tilde=float(s_tilde[k]),
                    ava_netted=float(model.kappa * abs(m[k]) * s_tilde[k]),
                    ava_addup=float(model.kappa * np.sum(np.abs(vega[mask]) * s[mask])),
                )
            )

        return SchemeEvaluation(
            scheme=self,
            te2=te2,
            var_total=var_total,
            r2=r2,
            alpha=alpha,
            budget=budget,
            passes_variance=te2 <= budget + 1e-12,
            ava=ava,
            ava_floor=model.ava_full,
            passes_floor=ava >= model.ava_full - 1e-9,
            ava_brut=model.ava_brut,
            set_stats=stats,
        )


@dataclass(frozen=True)
class SetStat:
    set_id: int
    size: int
    net_vega: float
    gross_vega: float
    s_tilde: float
    ava_netted: float
    ava_addup: float

    @property
    def offset_ratio(self) -> float:
        """1 - |net|/gross: how much signed vega cancels inside the set."""
        return 1.0 - abs(self.net_vega) / self.gross_vega if self.gross_vega > 0 else 0.0


@dataclass(frozen=True)
class SchemeEvaluation:
    scheme: NettingScheme
    te2: float
    var_total: float
    r2: float
    alpha: float
    budget: float
    passes_variance: bool
    ava: float
    ava_floor: float
    passes_floor: bool
    ava_brut: float
    set_stats: list[SetStat] = field(default_factory=list)

    @property
    def admissible(self) -> bool:
        return self.passes_variance and self.passes_floor

    @property
    def ava_saving(self) -> float:
        return self.ava_brut - self.ava

    @property
    def ava_saving_pct(self) -> float:
        return self.ava_saving / self.ava_brut if self.ava_brut > 0 else 0.0


# --------------------------------------------------------------------------- #
# Closed-form two-node case (Theoreme 2, sec. 5.2 of the note)
# --------------------------------------------------------------------------- #
def two_bucket(
    nu_i: float, nu_j: float, s_i: float, s_j: float, rho: float,
    var_total: float, alpha: float, kappa: float,
) -> dict:
    """Net node j onto pivot i: TE^2 (Th. 2 (i)), admissibility threshold
    rho_min (eq. 8), and the AVA gain (Th. 2 (iii)).

    ``var_total`` is the *portfolio* Var(DeltaPi) appearing on the RHS of
    the test; for an isolated pair use the pair's own variance.
    """
    te2 = nu_j ** 2 * (s_i ** 2 + s_j ** 2 - 2.0 * rho * s_i * s_j)
    rho_min = (
        (s_i ** 2 + s_j ** 2) / (2.0 * s_i * s_j)
        - (1.0 - alpha) * var_total / (2.0 * nu_j ** 2 * s_i * s_j)
    )
    s_tilde = s_i  # pivot shock
    gain = kappa * (abs(nu_i) * s_i + abs(nu_j) * s_j - abs(nu_i + nu_j) * s_tilde)
    return {
        "te2": float(te2),
        "budget": float((1.0 - alpha) * var_total),
        "rho_min": float(rho_min),
        "admissible": bool(rho >= rho_min),
        "ava_gain": float(gain),
        "net_vega": float(nu_i + nu_j),
    }
