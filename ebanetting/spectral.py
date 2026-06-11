"""Spectral diagnostic (sec. 6.1 of the note).

The covariance at the test nodes, Sigma_pil — the single statistical
input of the framework (sec. 4.1) — is diagonalised in an orthonormal
basis (spectral theorem): Sigma = sum_l lambda_l u_l u_l'. The *factors*
are the projections of the shock on the eigendirections,
xi_l = u_l' dsigma (Propriete 6: decorrelated, Var = lambda_l, exact
reconstruction, sum lambda = tr(Sigma) = sum s_i^2). With
c_l = <nu, u_l> the vega projection (Propriete 7):

    DeltaPi = sum_l c_l xi_l,    Var(DeltaPi) = sum_l c_l^2 lambda_l.

Theoreme 3 (spectral floor): for any netting scheme whose representative
shocks live in span(u_1..u_L) — the realistic description of coarse
schemes with smooth weights — TE^2 >= sum_{l>L} c_l^2 lambda_l: the tail
variance of the book is incompressible. (Without that hypothesis no
floor exists: w = nu replicates DeltaPi with one set — Remarque 5.)

Four usages (sec. 6.1.4): U1 inertia ratio tau_L (compressibility of the
uncertainty), U2 the risk map {c_l^2 lambda_l} (why a test fails and
where to refine), U3 spectral cleaning of a noisy Sigma-hat (flatten the
eigenvalue tail to its mean, with the conservatism guard), U4 stability
of the dominant subspaces across estimation windows. The limit: a factor
is NOT a netting set (dense signed weights — not a valuation exposure);
the spectrum measures, the partition nets.

Sigma_pil is assembled entrywise here (decoupled parameterisation of the
annex, scalar products — or the supplied corr_full); it is a small
(q x q) matrix on the pivot grid. No Kronecker / tensor product is used,
and the rest of the library never needs this matrix (quadratic forms go
through the matrix sandwich of Property 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .datasource import MarketDataBundle
from .model import UncertaintyModel

__all__ = [
    "node_covariance",
    "SpectralDiagnostic",
    "spectral_diagnostic",
    "spectral_clean",
    "subspace_stability",
]


def node_covariance(bundle: MarketDataBundle) -> np.ndarray:
    """Sigma_pil as a plain (n, n) symmetric matrix on the row-major node
    indexing i = a * qK + b — the note's statistical input.

    Entries are the scalar products of the annex's decoupled
    parameterisation, Sigma_ij = s_i s_j rho_mat[a,a'] rho_strike[b,b']
    (or the supplied corr_full). Assembled entrywise and finished with
    ordinary diagonal products D rho D — no Kronecker matrix product."""
    n, K = bundle.n, bundle.K
    if bundle.corr_full is not None:
        rho = np.asarray(bundle.corr_full, dtype=float)
    else:
        rho = np.empty((n, n))
        for i in range(n):
            a, b = divmod(i, K)
            for j in range(i, n):
                ap, bp = divmod(j, K)
                rho[i, j] = rho[j, i] = bundle.corr_mat[a, ap] * bundle.corr_strike[b, bp]
    d = np.diag(bundle.s.flatten())
    return d @ rho @ d


@dataclass(frozen=True)
class SpectralDiagnostic:
    eigenvalues: np.ndarray          # lambda_l, descending (Var of factor xi_l)
    projections: np.ndarray          # c_l = <nu, u_l> (Prop. 7)
    loadings2: np.ndarray            # c_l^2 lambda_l — the risk map (U2)
    residual_curve: np.ndarray       # residual_curve[L] = Th. 3 floor with L factors
    inertia: np.ndarray              # tau_L = sum_{l<=L} lambda / sum lambda (U1), L = 0..n
    var_total: float
    k_star: int                      # minimal L with floor <= budget
    alpha: float
    eigenvectors: np.ndarray         # columns u_l

    def explained_ratio(self, L: int) -> float:
        return 1.0 - self.residual_curve[L] / self.var_total if self.var_total > 0 else 1.0

    def mode(self, l: int, M: int, K: int) -> np.ndarray:
        """Eigendirection u_l displayed on the (tenor, strike) grid."""
        return self.eigenvectors[:, l].reshape(M, K)


def spectral_diagnostic(model: UncertaintyModel, alpha: float) -> SpectralDiagnostic:
    bundle = model.bundle
    sigma = node_covariance(bundle)
    lam, u = np.linalg.eigh(sigma)
    order = np.argsort(lam)[::-1]
    lam, u = np.clip(lam[order], 0.0, None), u[:, order]
    c = u.T @ bundle.vega.flatten()              # c_l = <nu, u_l>
    loadings2 = lam * c ** 2                     # Prop. 7: Var = sum c^2 lambda
    # residual_curve[L] = sum_{l > L} c_l^2 lambda_l (Theoreme 3)
    total = loadings2.sum()
    residual = np.concatenate([[total], total - loadings2.cumsum()])
    lam_total = lam.sum()
    inertia = np.concatenate([[0.0], lam.cumsum() / lam_total]) if lam_total > 0 else np.zeros(lam.size + 1)
    budget = model.budget(alpha)
    feasible = np.flatnonzero(residual <= budget + 1e-12)
    k_star = int(feasible[0]) if feasible.size else len(lam)
    return SpectralDiagnostic(
        eigenvalues=lam,
        projections=c,
        loadings2=loadings2,
        residual_curve=residual,
        inertia=inertia,
        var_total=model.var_total,
        k_star=k_star,
        alpha=alpha,
        eigenvectors=u,
    )


def spectral_clean(
    bundle: MarketDataBundle,
    L: Optional[int] = None,
    tau: float = 0.95,
) -> MarketDataBundle:
    """U3 — spectral cleaning of a noisy estimated covariance.

    On short histories the small eigenvalues are sampling noise (random
    matrix theory): kept as-is they hand the optimiser phantom
    correlations to exploit. Keep the dominant factors (the smallest L
    reaching inertia tau when L is not given) and flatten the eigenvalue
    tail to its mean — trace, hence total uncertainty sum s_i^2, is
    preserved (Prop. 6 (iii)). Returns a bundle carrying the cleaned
    covariance as corr_full.

    Conservatism guard (sec. 6.1.4): if the cleaned matrix increases the
    netting benefit, retain the lesser of the two — compare AVA figures
    on both bundles before adopting the cleaned one."""
    sigma = node_covariance(bundle)
    lam, u = np.linalg.eigh(sigma)
    order = np.argsort(lam)[::-1]
    lam, u = np.clip(lam[order], 0.0, None), u[:, order]
    if L is None:
        total = lam.sum()
        L = int(np.searchsorted(lam.cumsum() / total, tau) + 1) if total > 0 else lam.size
    L = max(1, min(int(L), lam.size))
    lam_clean = lam.copy()
    if L < lam.size:
        lam_clean[L:] = lam[L:].mean()
    sigma_clean = (u * lam_clean) @ u.T          # ordinary matrix products
    s_vec = np.sqrt(np.clip(np.diag(sigma_clean), 1e-12, None))
    corr = sigma_clean / s_vec[:, None] / s_vec[None, :]
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return MarketDataBundle(
        tenors=bundle.tenors,
        tenor_years=bundle.tenor_years,
        strikes=bundle.strikes,
        vega=bundle.vega,
        s=s_vec.reshape(bundle.M, bundle.K),
        corr_full=corr,
        meta={**bundle.meta, "spectral_clean": {"L": int(L), "tau": float(tau)}},
    )


def subspace_stability(
    bundle_a: MarketDataBundle,
    bundle_b: MarketDataBundle,
    L: int,
) -> np.ndarray:
    """U4 — stability of the dominant subspaces across estimation windows.

    Returns the principal-angle cosines between span(u_1..u_L) of the two
    covariances: the singular values of U_A' U_B (an ordinary (L, L)
    matrix product). All close to 1 means the structure of the
    uncertainty is stable in time — the upstream argument for why the
    bottom of the dendrogram (sec. 7.5) does not move between windows."""
    out = []
    for b in (bundle_a, bundle_b):
        lam, u = np.linalg.eigh(node_covariance(b))
        out.append(u[:, np.argsort(lam)[::-1][:L]])
    return np.linalg.svd(out[0].T @ out[1], compute_uv=False)
