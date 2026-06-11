"""Spectral lower bound on the achievable tracking error — complementary
diagnostic (not part of the current note, kept as model-validation aid).

Working in whitened coordinates W = N * s (diagonal scaling), Property 1
gives Var(DeltaPi) = <W, rho_mat W rho_strike>. With the per-axis
eigendecompositions rho_mat = U_m L_m U_m' and rho_strike = U_k L_k U_k'
(ordinary M x M and K x K problems), the variance decomposes exactly as

    Var(DeltaPi) = sum_{a,b} lam_m[a] lam_k[b] G[a,b]^2,
    G = U_m' W U_k          (ordinary matrix products),

so each 2-D mode (a, b) has eigenvalue lam_m[a] * lam_k[b], loading
G[a,b]^2, and grid pattern u_a @ u_b' — an (M,1)(1,K) matrix product.
For any representation using only L representative shocks the residual
variance is bounded below by the tail of that spectrum: the decay speed
dictates the minimal number of netting sets K*(alpha).

No Kronecker / tensor product is assembled at any point. When a full
``corr_full`` is supplied, its own (plain) eigendecomposition is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import UncertaintyModel

__all__ = ["SpectralDiagnostic", "spectral_diagnostic"]


@dataclass(frozen=True)
class SpectralDiagnostic:
    eigenvalues: np.ndarray          # lambda_l, descending
    loadings2: np.ndarray            # lambda_l * <W, mode_l>^2, same order
    residual_curve: np.ndarray       # residual_curve[L] = TE^2 lower bound with L factors
    var_total: float
    k_star: int                      # minimal L with residual <= budget
    alpha: float
    # decoupled case: per-axis eigenvectors + (a, b) index of each mode
    axis_vectors: Optional[tuple[np.ndarray, np.ndarray]] = None
    mode_pairs: Optional[np.ndarray] = None      # (n, 2) int
    # full-correlation case: plain eigenvectors of corr_full
    eigenvectors: Optional[np.ndarray] = None    # (n, n)

    def explained_ratio(self, L: int) -> float:
        return 1.0 - self.residual_curve[L] / self.var_total if self.var_total > 0 else 1.0

    def mode(self, l: int, M: int, K: int) -> np.ndarray:
        """Eigenmode l on the (tenor, strike) grid."""
        if self.axis_vectors is not None:
            u_m, u_k = self.axis_vectors
            a, b = self.mode_pairs[l]
            # u_a @ u_b' — an ordinary (M, 1)(1, K) matrix product
            return u_m[:, a].reshape(-1, 1) @ u_k[:, b].reshape(1, -1)
        return self.eigenvectors[:, l].reshape(M, K)


def spectral_diagnostic(model: UncertaintyModel, alpha: float) -> SpectralDiagnostic:
    bundle = model.bundle
    scaled = bundle.vega * bundle.s              # whitened vega W = N * s

    axis_vectors = None
    mode_pairs = None
    eigenvectors = None
    if bundle.is_decoupled():
        lam_m, u_m = np.linalg.eigh(bundle.corr_mat)
        lam_k, u_k = np.linalg.eigh(bundle.corr_strike)
        lam_m, lam_k = np.clip(lam_m, 0.0, None), np.clip(lam_k, 0.0, None)
        # 2-D eigenvalues lam_m[a] * lam_k[b] as a column @ row product
        lam2d = lam_m.reshape(-1, 1) @ lam_k.reshape(1, -1)
        g = u_m.T @ scaled @ u_k                 # ordinary matrix products
        load2d = lam2d * g ** 2
        order = np.argsort(lam2d, axis=None)[::-1]
        w = lam2d.flatten()[order]
        loadings2 = load2d.flatten()[order]
        pairs_a, pairs_b = np.unravel_index(order, lam2d.shape)
        mode_pairs = np.column_stack([pairs_a, pairs_b])
        axis_vectors = (u_m, u_k)
    else:
        w, v = np.linalg.eigh(bundle.corr_full)
        order = np.argsort(w)[::-1]
        w, v = np.clip(w[order], 0.0, None), v[:, order]
        loadings2 = w * (v.T @ scaled.flatten()) ** 2
        eigenvectors = v

    # residual_curve[L] = sum of loadings beyond the first L modes
    tail = np.concatenate([[loadings2.sum()], loadings2.cumsum()])
    residual = loadings2.sum() - tail[1:]
    residual = np.concatenate([[loadings2.sum()], residual])  # L = 0..n
    budget = model.budget(alpha)
    feasible = np.flatnonzero(residual <= budget + 1e-12)
    k_star = int(feasible[0]) if feasible.size else len(w)
    return SpectralDiagnostic(
        eigenvalues=w,
        loadings2=loadings2,
        residual_curve=residual,
        var_total=model.var_total,
        k_star=k_star,
        alpha=alpha,
        axis_vectors=axis_vectors,
        mode_pairs=mode_pairs,
        eigenvectors=eigenvectors,
    )
