"""Step 1 — passage from the system grid to the test nodes (sec. 3 of the
note).

Consensus uncertainty (Totem dispersions, broker ranges) is observable
only at a few pillars. The granular vega N (M, K) is transported onto the
pillar grid with two passage matrices (Def. 1: non-negative rows summing
to one) and the 2-D sandwich (Def. 2) — ordinary matrix products:

    N_tilde = A_T' N A_K   in R^{qT x qK}.

Three row-filling conventions (sec. 3.2): ``quadrant`` (all weight on the
nearest pillar), ``equal`` (1/2-1/2 on the bracketing pillars) and
``interp`` (linear interpolation weights, flat extrapolation outside the
pillar range). Theoreme 1: when the passage weights coincide with the
interpolation weights of the surface construction, the passage destroys
no information (TE_passage = 0); any other convention has a quantifiable
tracking error (eq. 5) charged against the variance-test budget.

Sec. 3.5 generalises to shock families whose grid differs from the sensi
grid: shocks *coarser* than the sensitivities are handled by the
contraction above (regime i); shocks *finer* (e.g. daily variations of
the full system surface vs the parametrisation pillars) are handled by
*restriction without loss* (Propriete 5): 0/1 selection matrices
R_T, R_K read the fine shock surface at the pillars,
dsigma_pil = R_T dsigma_fin R_K' — exact when the fine grid is generated
by (and contains) the pillars. The out-of-grid component
dsigma_fin - B_T (R_T dsigma_fin R_K') B_K' (Remarque 4) measures daily
how much surface movement the pillar set cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .datasource import MarketDataBundle

PassageConvention = Literal["interp", "equal", "quadrant"]

__all__ = [
    "PassageConvention",
    "PassageResult",
    "passage_matrix",
    "project",
    "project_bundle",
    "restriction_matrix",
    "restrict_shocks",
    "offgrid_residual",
]


def passage_matrix(
    grid: Sequence[float],
    pillars: Sequence[float],
    convention: PassageConvention = "interp",
) -> np.ndarray:
    """1-D passage matrix A (len(grid), len(pillars)) — Def. 1 of the note.

    Each row is non-negative and sums to one. Buckets outside the pillar
    range get flat extrapolation (all weight on the extreme pillar);
    buckets falling on a pillar map entirely to it.
    """
    grid = np.asarray(grid, dtype=float)
    pil = np.asarray(pillars, dtype=float)
    if pil.size < 1:
        raise ValueError("at least one pillar is required")
    if np.any(np.diff(pil) <= 0):
        raise ValueError("pillars must be strictly increasing")
    A = np.zeros((grid.size, pil.size))
    for i, t in enumerate(grid):
        if t <= pil[0]:
            A[i, 0] = 1.0
        elif t >= pil[-1]:
            A[i, -1] = 1.0
        else:
            b = int(np.searchsorted(pil, t))
            a = b - 1
            if pil[b] == t:
                A[i, b] = 1.0
            elif convention == "interp":
                wb = (t - pil[a]) / (pil[b] - pil[a])
                A[i, a], A[i, b] = 1.0 - wb, wb
            elif convention == "equal":
                A[i, a] = A[i, b] = 0.5
            elif convention == "quadrant":
                A[i, a if t - pil[a] <= pil[b] - t else b] = 1.0
            else:
                raise ValueError(f"unknown passage convention '{convention}'")
    return A


def project(vega: np.ndarray, a_t: np.ndarray, a_k: np.ndarray) -> np.ndarray:
    """2-D passage sandwich (Def. 2): N_tilde = A_T' N A_K."""
    return a_t.T @ np.asarray(vega, dtype=float) @ a_k


@dataclass(frozen=True)
class PassageResult:
    """Projected bundle plus the audit quantities of sec. 3."""

    node_bundle: MarketDataBundle      # everything downstream runs on this
    a_t: np.ndarray                    # (M, qT) passage matrix, tenor axis
    a_k: np.ndarray                    # (K, qK) passage matrix, strike axis
    convention: str
    vega_total_granular: float         # Property 3: both totals must match
    vega_total_nodes: float
    te2_passage: float                 # eq. (5), 0 for interp (Theoreme 1)
    te2_by_convention: dict            # convention -> TE^2 vs interp weights
    var_reference: float               # Var(<B_T' N B_K, dsigma_pil>), Th. 1

    @property
    def conserves_vega(self) -> bool:
        scale = max(abs(self.vega_total_granular), 1.0)
        return abs(self.vega_total_granular - self.vega_total_nodes) <= 1e-9 * scale


def project_bundle(
    bundle: MarketDataBundle,
    tenor_pillars: Sequence[int],
    strike_pillars: Sequence[int],
    convention: PassageConvention = "interp",
) -> PassageResult:
    """Project a granular bundle onto test nodes (pillars given as indices
    into the bundle's tenor / strike grids).

    The vega is contracted with the sandwich of Def. 2; the uncertainty
    model (s and correlations) is restricted to the pillars, where it is
    actually measured. The passage tracking error of every convention is
    reported against the interpolation weights (Theoreme 1 reference).
    """
    t_idx = np.asarray(sorted(tenor_pillars), dtype=int)
    k_idx = np.asarray(sorted(strike_pillars), dtype=int)
    if t_idx.size < 1 or k_idx.size < 1:
        raise ValueError("at least one pillar per axis is required")

    ten = np.asarray(bundle.tenor_years, dtype=float)
    stk = np.asarray(bundle.strikes, dtype=float)

    mats = {
        conv: (
            passage_matrix(ten, ten[t_idx], conv),
            passage_matrix(stk, stk[k_idx], conv),
        )
        for conv in ("interp", "equal", "quadrant")
    }
    a_t, a_k = mats[convention]
    vega_nodes = project(bundle.vega, a_t, a_k)

    s_nodes = bundle.s[np.ix_(t_idx, k_idx)]
    corr_mat = None if bundle.corr_mat is None else bundle.corr_mat[np.ix_(t_idx, t_idx)]
    corr_strike = None if bundle.corr_strike is None else bundle.corr_strike[np.ix_(k_idx, k_idx)]
    corr_full = None
    if bundle.corr_full is not None:
        flat = np.array([m * bundle.K + k for m in t_idx for k in k_idx])
        corr_full = bundle.corr_full[np.ix_(flat, flat)]

    node_bundle = MarketDataBundle(
        tenors=[bundle.tenors[i] for i in t_idx],
        tenor_years=[bundle.tenor_years[i] for i in t_idx],
        strikes=[bundle.strikes[i] for i in k_idx],
        vega=vega_nodes,
        s=s_nodes,
        corr_mat=corr_mat,
        corr_strike=corr_strike,
        corr_full=corr_full,
        meta={
            **bundle.meta,
            "passage": {
                "convention": convention,
                "tenor_pillars": [bundle.tenors[i] for i in t_idx],
                "strike_pillars": [bundle.strikes[i] for i in k_idx],
            },
        },
    )

    # Tracking error of each convention against the interpolation weights
    # (eq. 5): E = B_T' N B_K - A_T' N A_K, TE^2 = Var(<E, dsigma_pil>).
    b_t, b_k = mats["interp"]
    reference = project(bundle.vega, b_t, b_k)
    te2_by_convention = {
        conv: max(node_bundle.variance_of(reference - project(bundle.vega, m_t, m_k)), 0.0)
        for conv, (m_t, m_k) in mats.items()
    }

    return PassageResult(
        node_bundle=node_bundle,
        a_t=a_t,
        a_k=a_k,
        convention=convention,
        vega_total_granular=float(bundle.vega.sum()),
        vega_total_nodes=float(vega_nodes.sum()),
        te2_passage=te2_by_convention[convention],
        te2_by_convention=te2_by_convention,
        var_reference=max(node_bundle.variance_of(reference), 0.0),
    )


# --------------------------------------------------------------------------- #
# Sec. 3.5 — regime (ii): shocks finer than the sensitivities
# --------------------------------------------------------------------------- #
def restriction_matrix(fine_grid: Sequence[float], pillars: Sequence[float]) -> np.ndarray:
    """0/1 selection matrix R (q, P) of Propriete 5 — one entry 1 per row,
    picking each pillar's position in the fine grid.

    Requires every pillar to be present in the fine grid (the
    interpolation reproduces its nodes)."""
    fine = np.asarray(fine_grid, dtype=float)
    pil = np.asarray(pillars, dtype=float)
    R = np.zeros((pil.size, fine.size))
    for a, p in enumerate(pil):
        hits = np.flatnonzero(np.isclose(fine, p))
        if hits.size == 0:
            raise ValueError(f"pillar {p} is not a point of the fine grid")
        R[a, hits[0]] = 1.0
    return R


def restrict_shocks(shocks_fine: np.ndarray, r_t: np.ndarray, r_k: np.ndarray) -> np.ndarray:
    """Read a fine shock surface at the pillars (Propriete 5):
    dsigma_pil = R_T dsigma_fin R_K' — ordinary matrix products, exact
    (no representation error) when the fine surface is generated by the
    pillars."""
    return r_t @ np.asarray(shocks_fine, dtype=float) @ r_k.T


def offgrid_residual(
    shocks_fine: np.ndarray,
    b_t: np.ndarray,
    b_k: np.ndarray,
    r_t: np.ndarray,
    r_k: np.ndarray,
) -> np.ndarray:
    """Out-of-grid movement (Remarque 4):
    dsigma_fin - B_T (R_T dsigma_fin R_K') B_K'.

    Zero when the fine variation is generated by the pillars; its size,
    monitored daily, measures empirically the adequacy of the pillar set.
    The non-zero part is off-grid risk (shape-parameter sensitivities),
    to be treated separately — not a defect of the passage."""
    fine = np.asarray(shocks_fine, dtype=float)
    return fine - b_t @ (r_t @ fine @ r_k.T) @ b_k.T
