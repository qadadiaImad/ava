"""Vega netting engine (implementation sheet, volet 2/2).

Production model:  Cov = SigmaT . SigmaK + diag(sigma_eps^2)  — separable
factor matrices on the pillar grid plus a per-cell idiosyncratic map —
and the 2-D matrix sandwich  X_t = Y B_t Z' + E_t  with orthonormalised
polynomial bases Y (tenor axis) and Z (strike axis). Everything below is
ordinary matrix products, traces and reductions; matrices stay small
(<= 30 x 30) and nothing Kronecker-shaped is ever assembled.

Modules of the sheet:
  adapters  — bases (Gram-Schmidt on the grid, optional torsion mode),
              winsorisation, stale-cell detection;
  model     — flip-flop separable estimation (closed-form alternating
              maximisation with scale normalisation, ridge and shrinkage
              fallback), daily sandwich B_t = (Y'Y)^-1 Y' X_t Z (Z'Z)^-1,
              idio map, factor covariance Sigma_B, and the T1-T6 test
              battery with its automatic decisions;
  structure — adverse stress (shrink each factor matrix towards its
              diagonal, inflate the idio), closed-form cell distances
              d^2 = (ST)aa (SK)jj + (ST)bb (SK)ll - 2 (ST)ab (SK)jl
              + eps_aj^2 + eps_bl^2, agglomerative clustering (most
              liquid cell as pivot, contiguous rectangles, idio-dominant
              cells excluded), portfolio-free criterion, and the three
              stability metrics (survival frequency, adjusted Rand
              index, principal angles);
  evaluate  — the daily run E1-E6: exposure G = Y' N Z, true variance
              sum G G Sigma_B + sum N^2 eps^2, ONE global gap pattern
              A = N - N_rep, exact TE^2 = tr(A' SigmaT A SigmaK)
              + sum A^2 eps^2 (the trace handles every inter-group
              compensation), the always-logged triangle majorant, the
              GA = Y' A Z extraction diagnostic, AVA and floor;
  select    — cut sweep on a reference book (frozen monthly, never
              re-optimised per book) and the minimum-benefit guard
              across shock families.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .optimizer import Rect, _labels_from_rects, _mergeable, _union

__all__ = [
    "orthonormal_basis",
    "winsorize_panel",
    "stale_cells",
    "flip_flop",
    "EngineModel",
    "fit_engine_model",
    "stress_engine_model",
    "cell_liquidity",
    "cell_distances",
    "engine_dendrogram",
    "cut_engine",
    "adjusted_rand_index",
    "principal_angle_cosines",
    "survival_frequencies",
    "EngineRun",
    "evaluate_book_engine",
    "select_cut",
    "min_benefit_guard",
    "simulate_panel",
]


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def orthonormal_basis(coords: Sequence[float], degree: int = 2,
                      torsion: bool = False) -> np.ndarray:
    """Polynomial basis [1, c, c^2, ...] on the grid, orthonormalised by
    discrete Gram-Schmidt (QR) so the sandwich factors decouple. With
    ``torsion`` an extra odd-even crossed column |c| is appended — the
    kink mode: a slope differing between the two wings is a x + b |x|,
    so |c| is the wing-asymmetry factor of the T4 automatic decision."""
    c = np.asarray(coords, dtype=float)
    cols = [c ** p for p in range(degree + 1)]
    if torsion:
        cols.append(np.abs(c))
    raw = np.column_stack(cols)
    q, r = np.linalg.qr(raw)
    # fix signs so each mode correlates positively with its raw column
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def winsorize_panel(panel: np.ndarray, q: float = 0.005) -> np.ndarray:
    """Per-cell winsorisation of the daily variations at the q / 1-q
    quantiles (adapters point 3c)."""
    panel = np.asarray(panel, dtype=float)
    lo = np.quantile(panel, q, axis=0)
    hi = np.quantile(panel, 1.0 - q, axis=0)
    return np.clip(panel, lo, hi)


def stale_cells(panel: np.ndarray, max_flat: int = 5) -> np.ndarray:
    """(M, K) boolean mask of cells showing a zero variation for more
    than ``max_flat`` consecutive days (adapters point 3b) — to be
    flagged stale and excluded from the estimation."""
    panel = np.asarray(panel, dtype=float)
    flat = panel == 0.0
    run = np.zeros(panel.shape[1:], dtype=int)
    worst = np.zeros_like(run)
    for t in range(panel.shape[0]):
        run = np.where(flat[t], run + 1, 0)
        worst = np.maximum(worst, run)
    return worst > max_flat


# --------------------------------------------------------------------------- #
# model — flip-flop separable estimation
# --------------------------------------------------------------------------- #
def flip_flop(panel: np.ndarray, tol: float = 1e-6, max_iter: int = 50,
              ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray, int]:
    """Alternating closed-form maximisation of the separable covariance
    (sec. 3.1): SigmaT = (1/nK) sum X SigmaK^-1 X', then SigmaK =
    (1/nM) sum X' SigmaT^-1 X, iterated to convergence, then the scale
    indeterminacy is fixed by SigmaK <- K SigmaK / tr(SigmaK)."""
    panel = np.asarray(panel, dtype=float)
    n, M, K = panel.shape
    sigma_k = np.eye(K)
    sigma_t = np.eye(M)
    for it in range(1, max_iter + 1):
        k_inv = np.linalg.inv(sigma_k + ridge * np.trace(sigma_k) / K * np.eye(K))
        new_t = sum(x @ k_inv @ x.T for x in panel) / (n * K)
        t_inv = np.linalg.inv(new_t + ridge * np.trace(new_t) / M * np.eye(M))
        new_k = sum(x.T @ t_inv @ x for x in panel) / (n * M)
        moved = (
            np.linalg.norm(new_t - sigma_t) / max(np.linalg.norm(new_t), 1e-300)
            + np.linalg.norm(new_k - sigma_k) / max(np.linalg.norm(new_k), 1e-300)
        )
        sigma_t, sigma_k = new_t, new_k
        if moved < tol:
            break
    scale = np.trace(sigma_k) / K
    return sigma_t * scale, sigma_k / scale, it


def _shrink_to_diagonal(sigma: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * sigma + weight * np.diag(np.diag(sigma))


@dataclass(frozen=True)
class EngineModel:
    """Output of the monthly model run (sec. 3): estimated production
    model + sandwich material + the T1-T6 verdicts."""

    sigma_t: np.ndarray            # (M, M) tenor factor covariance
    sigma_k: np.ndarray            # (K, K) strike factor covariance
    sigma_eps: np.ndarray          # (M, K) idio map, per cell
    y: np.ndarray                  # (M, Fy) orthonormal tenor basis
    z: np.ndarray                  # (K, Fz) orthonormal strike basis
    sigma_b: np.ndarray            # (Fy*Fz, Fy*Fz) factor covariance
    b_sd: np.ndarray               # (Fy, Fz) named factor sds (skew = [0,1]...)
    r2: np.ndarray                 # (T,) daily sandwich explanatory power
    tests: dict = field(default_factory=dict)
    majorant_only: bool = False    # automatic decision on T1/T5 failure
    idio_dominant: Optional[np.ndarray] = None   # (M, K) bool, T6 exclusion list

    @property
    def shape(self) -> tuple[int, int]:
        return self.sigma_t.shape[0], self.sigma_k.shape[0]

    def variance_of(self, exposure: np.ndarray) -> float:
        """Var(<A, X>) under the production model — the trace formula:
        tr(A' SigmaT A SigmaK) + sum A^2 eps^2 (ordinary products)."""
        a = np.asarray(exposure, dtype=float)
        return float(np.trace(a.T @ self.sigma_t @ a @ self.sigma_k)
                     + np.sum(a ** 2 * self.sigma_eps ** 2))


def _run_tests(panel, model: EngineModel, alpha: float, reference_book,
               pair_samples: int, rng) -> tuple[dict, bool, np.ndarray]:
    """T1-T6 battery (sec. 3.3) with the sheet's default thresholds."""
    M, K = model.shape
    y, z = model.y, model.z
    tests = {}
    # T1 — explanatory power of the sandwich
    tests["T1"] = {
        "mean_r2": float(model.r2.mean()),
        "q10_r2": float(np.quantile(model.r2, 0.10)),
        "passed": bool(model.r2.mean() >= 0.85 and np.quantile(model.r2, 0.10) >= 0.70),
    }
    # T2 — mode shapes: leading eigenvectors of each small matrix project
    # on the orthonormal polynomial basis of its axis
    def alignment(sigma, basis):
        lam, u = np.linalg.eigh(sigma)
        u = u[:, np.argsort(lam)[::-1]]
        return [float(np.sum((basis.T @ u[:, l]) ** 2)) for l in range(min(3, u.shape[1]))]
    align_t, align_k = alignment(model.sigma_t, y), alignment(model.sigma_k, z)
    tests["T2"] = {
        "alignment_tenor": align_t,
        "alignment_strike": align_k,
        "passed": bool(min(align_t[:2]) >= 0.90 and min(align_k[:2]) >= 0.90),
    }
    # T3 — regression / spectral coherence: under separability the 2-D
    # share of the 9-factor span is the product of the per-axis inertias
    # tau_3; it must agree with the sandwich's mean R^2
    def tau3(sigma):
        lam = np.sort(np.linalg.eigvalsh(sigma))[::-1]
        return float(lam[:3].sum() / lam.sum()) if lam.sum() > 0 else 1.0
    gap = abs(tau3(model.sigma_t) * tau3(model.sigma_k) - tests["T1"]["mean_r2"])
    tests["T3"] = {"max_gap": gap, "passed": bool(gap <= 0.05)}
    # T4 — idio really idio: off-diagonal mass of the residual column
    # correlations + the torsion test (slope on the put wing vs call wing)
    resid = panel - np.stack([y @ b @ z.T for b in _b_series(panel, y, z)])
    flat = resid.reshape(panel.shape[0], -1)
    sd = flat.std(axis=0)
    keep = sd > 1e-12
    corr = np.corrcoef(flat[:, keep].T)
    mass = float((np.abs(corr).sum() - np.trace(np.abs(corr)))
                 / max(corr.shape[0] ** 2 - corr.shape[0], 1))
    x = z[:, 1]  # orthonormal slope coordinate
    # per-tranche, per-date: remove the full-grid fit of every NON-slope
    # component (level, curvature, kink when present — well-conditioned,
    # all K points), then fit the slope on each wing of the cleaned row.
    # A wing-restricted multi-column solve would amplify the idio by the
    # conditioning of clustered points; cleaning first avoids it, and a
    # kink mode present in the basis is removed here — so once the
    # automatic decision has added it, the wing slopes agree again.
    pz = np.linalg.pinv(z)
    non_slope = [l for l in range(z.shape[1]) if l != 1]
    torsions = []
    for a in range(M):
        rows = panel[:, a, :]                       # (T, K)
        coef_full = pz @ rows.T                     # (Fz, T)
        cleaned = rows - (z[:, non_slope] @ coef_full[non_slope]).T
        for_wings = []
        for w in (x < 0, x > 0):
            if w.sum() < 2:
                for_wings = []
                break
            xc = x[w] - x[w].mean()
            sub = cleaned[:, w]
            slope = (sub - sub.mean(axis=1, keepdims=True)) @ xc / float(xc @ xc)
            for_wings.append(slope)
        if for_wings and for_wings[0].std() > 1e-12 and for_wings[1].std() > 1e-12:
            torsions.append(float(np.corrcoef(for_wings[0], for_wings[1])[0, 1]))
    torsion = float(np.mean(torsions)) if torsions else 1.0
    tests["T4"] = {
        "offdiag_mass": mass,
        "torsion_corr": torsion,
        "passed": bool(mass <= 0.10 and torsion >= 0.80),
        "torsion_passed": bool(torsion >= 0.80),
    }
    # T5 — separability OF THE DATA: sampled pairwise empirical
    # covariances vs the best separable surrogate. The surrogate is
    # moment-matched (no matrix inversion):
    #   (1/n) sum X X' - diag(sum_j eps^2)  =  SigmaT . tr(SigmaK),
    #   (1/n) sum X'X  - diag(sum_a eps^2)  =  SigmaK . tr(SigmaT),
    # so the model pair covariance is S_T[a,b] S_K[j,l] / C with
    # C = total common variance — this keeps T5 a test of the data,
    # immune to the flip-flop's behaviour on near-singular factors.
    eps2 = model.sigma_eps ** 2
    s_t = sum(x @ x.T for x in panel) / panel.shape[0] - np.diag(eps2.sum(axis=1))
    s_k = sum(x.T @ x for x in panel) / panel.shape[0] - np.diag(eps2.sum(axis=0))
    c_tot = max(float(np.mean(panel ** 2, axis=0).sum() - eps2.sum()), 1e-300)
    idx = rng.choice(M * K, size=(pair_samples, 2))
    num = den = 0.0
    flat_panel = panel.reshape(panel.shape[0], -1)
    for i, j in idx:
        a, jj = divmod(int(i), K)
        b, ll = divmod(int(j), K)
        emp = float(np.cov(flat_panel[:, i], flat_panel[:, j])[0, 1])
        mod = s_t[a, b] * s_k[jj, ll] / c_tot
        if i == j:
            mod += eps2[a, jj]
        num += (emp - mod) ** 2
        den += emp ** 2
    err = float(np.sqrt(num / max(den, 1e-300)))
    tests["T5"] = {"rel_error": err, "passed": bool(err <= 0.15)}
    # T6 — idio negligibility for a reference book
    ref = np.ones((M, K)) if reference_book is None else np.asarray(reference_book, float)
    budget = (1.0 - alpha) * model.variance_of(ref)
    contrib = ref ** 2 * model.sigma_eps ** 2
    ratio = float(contrib.sum() / budget) if budget > 0 else np.inf
    dominant = contrib > 0.05 * budget
    tests["T6"] = {
        "idio_budget_ratio": ratio,
        "n_dominant_cells": int(dominant.sum()),
        "passed": bool(ratio <= 0.15),
    }
    majorant_only = not (tests["T1"]["passed"] and tests["T5"]["passed"])
    return tests, majorant_only, dominant


def _b_series(panel: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Daily sandwich coefficients B_t = (Y'Y)^-1 Y' X_t Z (Z'Z)^-1 —
    with orthonormal bases the projectors reduce to Y' X Z."""
    py = np.linalg.inv(y.T @ y) @ y.T
    pz = z @ np.linalg.inv(z.T @ z)
    return np.stack([py @ x @ pz for x in panel])


def fit_engine_model(
    panel: np.ndarray,
    tenor_coords: Sequence[float],
    strike_coords: Sequence[float],
    alpha: float = 0.95,
    torsion: bool = False,
    reference_book: Optional[np.ndarray] = None,
    min_days_margin: int = 10,
    pair_samples: int = 200,
    seed: int = 0,
) -> EngineModel:
    """Monthly model run (sec. 3): flip-flop + sandwich + idio map +
    Sigma_B + the T1-T6 battery and its automatic decisions. When the
    panel is too short (n < max(M, K) + margin) each small matrix is
    shrunk towards its diagonal (Ledoit-Wolf-style fallback). The
    T4-torsion automatic decision re-estimates with the extra crossed
    mode (the sandwich grows to 12 factors)."""
    panel = np.asarray(panel, dtype=float)
    n, M, K = panel.shape
    # coordinates of the sheet: x = moneyness fwd - 1, y = log T standardised
    ten = np.log(np.asarray(tenor_coords, dtype=float))
    ten = (ten - ten.mean()) / max(ten.std(), 1e-12)
    stk = np.asarray(strike_coords, dtype=float) - 1.0
    y = orthonormal_basis(ten)
    z = orthonormal_basis(stk, torsion=torsion)

    sigma_t, sigma_k, _ = flip_flop(panel)
    if n < max(M, K) + min_days_margin:
        w = min(1.0, (max(M, K) + min_days_margin - n) / max(M, K))
        sigma_t = _shrink_to_diagonal(sigma_t, w)
        sigma_k = _shrink_to_diagonal(sigma_k, w)

    b = _b_series(panel, y, z)
    fitted = np.stack([y @ bt @ z.T for bt in b])
    resid = panel - fitted
    # idio map: residual std corrected for the projection's leverages
    # (the raw residual underestimates eps by sqrt(1 - h_a h_j))
    h_y = np.diag(y @ np.linalg.inv(y.T @ y) @ y.T)
    h_z = np.diag(z @ np.linalg.inv(z.T @ z) @ z.T)
    deflate = np.sqrt(np.clip(1.0 - h_y[:, None] * h_z[None, :], 0.05, None))
    sigma_eps = resid.std(axis=0) / deflate
    denom = np.sum(panel.reshape(n, -1) ** 2, axis=1)
    r2 = 1.0 - np.sum(resid.reshape(n, -1) ** 2, axis=1) / np.where(denom > 0, denom, 1.0)
    flat_b = b.reshape(n, -1)
    sigma_b = np.cov(flat_b.T) if n > 1 else np.zeros((flat_b.shape[1],) * 2)
    b_sd = flat_b.std(axis=0).reshape(b.shape[1], b.shape[2])

    model = EngineModel(
        sigma_t=sigma_t, sigma_k=sigma_k, sigma_eps=sigma_eps,
        y=y, z=z, sigma_b=sigma_b, b_sd=b_sd, r2=r2,
    )
    rng = np.random.default_rng(seed)
    tests, majorant_only, dominant = _run_tests(
        panel, model, alpha, reference_book, pair_samples, rng)
    if not torsion and not tests["T4"]["passed"]:
        # automatic decision: try the crossed odd-even (kink) mode and
        # re-estimate; ADOPT it only if it materially improves the
        # sandwich's explanatory power — a genuine torsion does, mere
        # out-of-span noise does not (no spurious factor inflation)
        refit = fit_engine_model(panel, tenor_coords, strike_coords, alpha=alpha,
                                 torsion=True, reference_book=reference_book,
                                 min_days_margin=min_days_margin,
                                 pair_samples=pair_samples, seed=seed)
        if refit.tests["T1"]["mean_r2"] >= tests["T1"]["mean_r2"] + 0.02:
            return refit
    return replace(model, tests=tests, majorant_only=majorant_only,
                   idio_dominant=dominant)


# --------------------------------------------------------------------------- #
# structure — underlying layer
# --------------------------------------------------------------------------- #
def stress_engine_model(model: EngineModel, delta: float = 0.2,
                        idio_mult: float = 1.2) -> EngineModel:
    """Adverse stress (sec. 4.1): pull each small factor matrix towards
    its diagonal — every disagreement increases — and inflate the idio."""
    return replace(
        model,
        sigma_t=_shrink_to_diagonal(model.sigma_t, delta),
        sigma_k=_shrink_to_diagonal(model.sigma_k, delta),
        sigma_eps=idio_mult * model.sigma_eps,
    )


def cell_liquidity(model: EngineModel) -> np.ndarray:
    """s_aj = sqrt((SigmaT)aa (SigmaK)jj + eps_aj^2) — the pivot of a
    group is its most liquid cell (smallest s)."""
    dt = np.diag(model.sigma_t).reshape(-1, 1)
    dk = np.diag(model.sigma_k).reshape(1, -1)
    return np.sqrt(dt @ dk + model.sigma_eps ** 2)


def cell_distances(model: EngineModel) -> np.ndarray:
    """Closed-form disagreement between cells (a, j) and (b, l):
    d^2 = (ST)aa (SK)jj + (ST)bb (SK)ll - 2 (ST)ab (SK)jl
          + eps_aj^2 + eps_bl^2,  as an (n, n) distance table."""
    M, K = model.shape
    n = M * K
    d2 = np.zeros((n, n))
    st, sk, eps = model.sigma_t, model.sigma_k, model.sigma_eps
    for i in range(n):
        a, j = divmod(i, K)
        for jdx in range(i + 1, n):
            b, l = divmod(jdx, K)
            v = (st[a, a] * sk[j, j] + st[b, b] * sk[l, l]
                 - 2.0 * st[a, b] * sk[j, l]
                 + eps[a, j] ** 2 + eps[b, l] ** 2)
            d2[i, jdx] = d2[jdx, i] = max(v, 0.0)
    return np.sqrt(d2)


@dataclass(frozen=True)
class EngineMerge:
    step: int
    rect_a: Rect
    rect_b: Rect
    merged: Rect
    pivot: tuple            # most liquid cell of the merged group
    height: float           # pivot-to-pivot disagreement d
    portfolio_free: bool    # d_{j,pivot} <= eps * s_j for every member


def _rect_cells(rect: Rect):
    r0, r1, c0, c1 = rect
    return [(m, k) for m in range(r0, r1 + 1) for k in range(c0, c1 + 1)]


def engine_dendrogram(model: EngineModel, epsilon: float = 0.30,
                      stressed: bool = True, delta: float = 0.2,
                      idio_mult: float = 1.2) -> list[EngineMerge]:
    """Structure run (sec. 4): stress, distances, then agglomerative
    clustering of contiguous rectangles by increasing pivot-linkage
    disagreement; idio-dominant cells (T6) are never merged; every merge
    carries its portfolio-free admissibility flag (d <= eps * s)."""
    work = stress_engine_model(model, delta, idio_mult) if stressed else model
    M, K = work.shape
    dist = cell_distances(work)
    s = cell_liquidity(work)
    excluded = model.idio_dominant if model.idio_dominant is not None else np.zeros((M, K), bool)

    rects: list[Rect] = [(m, m, k, k) for m in range(M) for k in range(K)]
    pivots = {r: _rect_cells(r)[0] for r in rects}
    merges: list[EngineMerge] = []
    while len(rects) > 1:
        best = None
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if not _mergeable(rects[i], rects[j]):
                    continue
                if any(excluded[c] for c in _rect_cells(rects[i]) + _rect_cells(rects[j])):
                    continue
                pa, pb = pivots[rects[i]], pivots[rects[j]]
                h = float(dist[pa[0] * K + pa[1], pb[0] * K + pb[1]])
                if best is None or h < best[0]:
                    best = (h, i, j)
        if best is None:
            break
        h, i, j = best
        union = _union(rects[i], rects[j])
        cells = _rect_cells(union)
        pivot = cells[int(np.argmin([s[c] for c in cells]))]
        ip = pivot[0] * K + pivot[1]
        pf = all(dist[c[0] * K + c[1], ip] <= epsilon * s[c] for c in cells)
        merges.append(EngineMerge(step=len(merges) + 1, rect_a=rects[i],
                                  rect_b=rects[j], merged=union,
                                  pivot=pivot, height=h, portfolio_free=pf))
        rects = [r for idx, r in enumerate(rects) if idx not in (i, j)] + [union]
        pivots = {r: pivots.get(r, pivot) for r in rects}
        pivots[union] = pivot
    return merges


def cut_engine(merges: list[EngineMerge], M: int, K: int,
               height: float, require_portfolio_free: bool = False) -> np.ndarray:
    """(M, K) labels of the structure at the given cut height — merges
    applied in order while height <= cut. The portfolio-free flag MARKS
    each fusion (sec. 4.1 point 4: the TE <= eps * add-up bound holds for
    any book on flagged fusions — evidence material); requiring it makes
    the cut strictly portfolio-free."""
    rects: list[Rect] = [(m, m, k, k) for m in range(M) for k in range(K)]
    for mg in merges:
        if mg.height > height:
            break
        if require_portfolio_free and not mg.portfolio_free:
            continue
        if mg.rect_a in rects and mg.rect_b in rects:
            rects = [r for r in rects if r not in (mg.rect_a, mg.rect_b)]
            rects.append(mg.merged)
    return _labels_from_rects(rects, M, K)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """ARI between two partitions (1 = identical, ~0 = independent) —
    the window-to-window stability metric of sec. 4.2."""
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    ua, ub = np.unique(a), np.unique(b)
    table = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua], float)
    n = a.size
    comb = lambda v: v * (v - 1) / 2.0
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(n)
    max_idx = 0.5 * (sum_a + sum_b)
    return float((sum_ij - expected) / (max_idx - expected)) if max_idx != expected else 1.0


def principal_angle_cosines(sigma_a: np.ndarray, sigma_b: np.ndarray, L: int = 3) -> np.ndarray:
    """Principal-angle cosines between the dominant L-eigenspaces of two
    small factor matrices (sec. 4.2 metric 3): all >= 0.95 expected when
    the modes do not rotate."""
    out = []
    for s in (sigma_a, sigma_b):
        lam, u = np.linalg.eigh(s)
        out.append(u[:, np.argsort(lam)[::-1][:L]])
    return np.linalg.svd(out[0].T @ out[1], compute_uv=False)


def survival_frequencies(
    panels: list[np.ndarray],
    tenor_coords: Sequence[float],
    strike_coords: Sequence[float],
    height: float,
    epsilon: float = 0.30,
    **fit_kwargs,
) -> dict:
    """Sec. 4.2 metric 1 — replay the whole chain (estimation -> stress ->
    dendrogram -> cut) on each panel (rolling windows and/or block
    bootstraps prepared by the caller) and count, for each candidate
    fusion (rectangle), its frequency of appearance under the cut.
    Fusions with frequency >= 90% are the retained, defendable ones."""
    counts: dict = {}
    for panel in panels:
        model = fit_engine_model(panel, tenor_coords, strike_coords, **fit_kwargs)
        merges = engine_dendrogram(model, epsilon=epsilon)
        M, K = model.shape
        labels = cut_engine(merges, M, K, height)
        for sid in np.unique(labels):
            cells = tuple(sorted(map(tuple, np.argwhere(labels == sid))))
            if len(cells) > 1:
                counts[cells] = counts.get(cells, 0) + 1
    return {k: v / len(panels) for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# evaluate — daily book run (E1-E6)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineRun:
    exposure: np.ndarray           # G = Y' N Z
    gap: np.ndarray                # A = N - N_rep, ONE global pattern
    te2: float                     # exact trace formula
    te2_majorant: float            # (sum |A| s)^2 — always logged
    budget: float
    var_total: float
    passes_variance: bool
    extractions: list              # [(k, l, |GA_kl|, sd, ava), ...]
    ava: float
    ava_brut: float
    ava_floor: float
    passes_floor: bool
    majorant_mode: bool            # the underlying is in majorant-only mode
    labels: np.ndarray
    pivots: dict

    @property
    def admissible(self) -> bool:
        return self.passes_variance and self.passes_floor


def _representation(n_mat: np.ndarray, labels: np.ndarray, s_map: np.ndarray):
    """N_rep: each group's vega summed onto its most liquid cell."""
    rep = np.zeros_like(n_mat, dtype=float)
    pivots = {}
    for sid in np.unique(labels):
        mask = labels == sid
        cells = np.argwhere(mask)
        pivot = tuple(cells[int(np.argmin([s_map[tuple(c)] for c in cells]))])
        rep[pivot] += float(n_mat[mask].sum())
        pivots[int(sid)] = pivot
    return rep, pivots


def evaluate_book_engine(
    n_mat: np.ndarray,
    model: EngineModel,
    labels: np.ndarray,
    alpha: float,
    kappa: float,
    max_extractions: int = 4,
) -> EngineRun:
    """The daily run E1-E6 of sec. 5.1, all in small matrices.

    E2  G = Y' N Z and the true variance sum G G Sigma_B + sum N^2 eps^2;
    E3  one global gap pattern A = N - N_rep — the trace formula handles
        every inter-group compensation, there is structurally no
        per-group variance accumulation to code;
    E4  exact TE^2 = tr(A' SigmaT A SigmaK) + sum A^2 eps^2, with the
        triangle majorant always computed and logged;
    E5  on failure, the GA = Y' A Z heatmap designates the dominant mode
        components, extracted in add-up (|GA_kl| sd_kl) as rank-one
        Y_k Z_l' subtractions from A, then re-test;
    E6  AVA = kappa sum |m_g| s_pivot + kappa sum extracted, floor."""
    n_mat = np.asarray(n_mat, dtype=float)
    labels = np.asarray(labels, dtype=int)
    s_map = cell_liquidity(model)

    g = model.y.T @ n_mat @ model.z                              # E2
    gvec = g.flatten()
    var_total = float(gvec @ model.sigma_b @ gvec
                      + np.sum(n_mat ** 2 * model.sigma_eps ** 2))
    budget = (1.0 - alpha) * var_total

    rep, pivots = _representation(n_mat, labels, s_map)          # E3
    gap = n_mat - rep
    majorant = float(np.sum(np.abs(gap) * s_map))                # E4

    extractions: list = []
    a = gap.copy()
    te2 = model.variance_of(a)
    if not model.majorant_only:                                  # E5
        while te2 > budget and len(extractions) < max_extractions:
            ga = model.y.T @ a @ model.z
            weights = np.abs(ga) * model.b_sd
            k, l = np.unravel_index(int(np.argmax(weights)), ga.shape)
            if weights[k, l] <= 0:
                break
            coeff = float(ga[k, l])
            extractions.append(
                (int(k), int(l), abs(coeff), float(model.b_sd[k, l]),
                 float(kappa * abs(coeff) * model.b_sd[k, l]))
            )
            # remove the component: rank-one Y_k Z_l' subtraction
            a = a - coeff * model.y[:, [k]] @ model.z[:, [l]].T
            te2 = model.variance_of(a)
    effective_te2 = majorant ** 2 if model.majorant_only else te2

    ava = float(kappa * sum(abs(n_mat[labels == sid].sum()) * s_map[p]
                            for sid, p in pivots.items()))
    ava += float(sum(e[4] for e in extractions))                 # E6
    ava_brut = float(kappa * np.sum(np.abs(n_mat) * s_map))
    ava_floor = float(kappa * np.sqrt(max(var_total, 0.0)))
    return EngineRun(
        exposure=g,
        gap=gap,
        te2=te2,
        te2_majorant=float(majorant ** 2),
        budget=budget,
        var_total=var_total,
        passes_variance=effective_te2 <= budget + 1e-12,
        extractions=extractions,
        ava=ava,
        ava_brut=ava_brut,
        ava_floor=ava_floor,
        passes_floor=ava >= ava_floor - 1e-9,
        majorant_mode=model.majorant_only,
        labels=labels,
        pivots=pivots,
    )


# --------------------------------------------------------------------------- #
# select — cut selection and the cross-family guard
# --------------------------------------------------------------------------- #
def select_cut(
    model: EngineModel,
    merges: list[EngineMerge],
    reference_book: np.ndarray,
    alpha: float,
    kappa: float,
) -> dict:
    """Sec. 6.1 — sweep the dendrogram cut heights on a REFERENCE book,
    trace the AVA-fidelity frontier and retain the admissible cut of
    minimal AVA. The cut is frozen per underlying at the monthly run:
    never re-optimised per book."""
    M, K = model.shape
    heights = sorted({0.0, *(m.height for m in merges)})
    frontier = []
    best = None
    for h in heights:
        labels = cut_engine(merges, M, K, h)
        run = evaluate_book_engine(reference_book, model, labels, alpha, kappa)
        point = {"height": h, "n_sets": int(labels.max()) + 1,
                 "ava": run.ava, "te2": run.te2, "budget": run.budget,
                 "admissible": run.admissible}
        frontier.append(point)
        if run.admissible and (best is None or run.ava < best["ava"]):
            best = point
    return {"frontier": frontier, "chosen": best}


def min_benefit_guard(runs_by_family: dict) -> tuple[str, dict]:
    """Sec. 6.2 — across shock families (daily, stressed daily, Totem...)
    never the 'best': retain the family giving the MINIMUM netting
    benefit (brut - netted). An estimation that over-correlates
    over-justifies the netting — the forbidden error direction."""
    name = min(runs_by_family,
               key=lambda k: runs_by_family[k].ava_brut - runs_by_family[k].ava)
    return name, runs_by_family[name]


# --------------------------------------------------------------------------- #
# golden-test material — synthetic generator (sec. 5.2)
# --------------------------------------------------------------------------- #
def simulate_panel(
    sigma_t: np.ndarray,
    sigma_k: np.ndarray,
    sigma_eps: np.ndarray,
    n_days: int,
    seed: int = 0,
) -> np.ndarray:
    """Simulate X_t under the production model: matrix-normal part
    SigmaT^(1/2) W SigmaK^(1/2) (ordinary matrix products on a white
    noise matrix) plus the per-cell idio."""
    rng = np.random.default_rng(seed)
    M, K = sigma_t.shape[0], sigma_k.shape[0]

    def sqrtm(s):
        lam, u = np.linalg.eigh(s)
        return (u * np.sqrt(np.clip(lam, 0.0, None))) @ u.T

    rt, rk = sqrtm(sigma_t), sqrtm(sigma_k)
    common = np.stack([rt @ rng.normal(size=(M, K)) @ rk for _ in range(n_days)])
    idio = rng.normal(size=(n_days, M, K)) * np.asarray(sigma_eps)[None, :, :]
    return common + idio
