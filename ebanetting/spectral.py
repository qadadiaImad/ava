"""Spectral (PCA) lower bound on the achievable tracking error
(Lemma 1, sec. 6.3 of the note).

For any rank-K linear representation, TE^2 >= sum_{l > K} lambda_l <nu, u_l>^2
where Sigma = sum_l lambda_l u_l u_l'. The decay of that tail, weighted by
the vega profile, dictates the minimal number of netting sets K*(alpha).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import UncertaintyModel

__all__ = ["SpectralDiagnostic", "spectral_diagnostic"]


@dataclass(frozen=True)
class SpectralDiagnostic:
    eigenvalues: np.ndarray          # lambda_l, descending
    eigenvectors: np.ndarray         # columns u_l
    loadings2: np.ndarray            # lambda_l * <nu, u_l>^2
    residual_curve: np.ndarray       # residual_curve[K] = TE^2 lower bound with K factors
    var_total: float
    k_star: int                      # minimal K with residual <= budget
    alpha: float

    def explained_ratio(self, K: int) -> float:
        return 1.0 - self.residual_curve[K] / self.var_total if self.var_total > 0 else 1.0

    def mode(self, l: int, M: int, K: int) -> np.ndarray:
        """Eigenmode l reshaped on the (tenor, strike) grid."""
        return self.eigenvectors[:, l].reshape(M, K)


def spectral_diagnostic(model: UncertaintyModel, alpha: float) -> SpectralDiagnostic:
    w, v = np.linalg.eigh(model.sigma)
    order = np.argsort(w)[::-1]
    w, v = np.clip(w[order], 0.0, None), v[:, order]
    proj2 = (v.T @ model.nu) ** 2
    loadings2 = w * proj2                      # variance carried by each mode
    # residual_curve[K] = sum of loadings beyond the first K modes
    tail = np.concatenate([[loadings2.sum()], loadings2.cumsum()])
    residual = loadings2.sum() - tail[1:]
    residual = np.concatenate([[loadings2.sum()], residual])  # K = 0..n
    budget = model.budget(alpha)
    feasible = np.flatnonzero(residual <= budget + 1e-12)
    k_star = int(feasible[0]) if feasible.size else len(w)
    return SpectralDiagnostic(
        eigenvalues=w,
        eigenvectors=v,
        loadings2=loadings2,
        residual_curve=residual,
        var_total=model.var_total,
        k_star=k_star,
        alpha=alpha,
    )
