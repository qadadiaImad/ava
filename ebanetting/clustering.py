"""Decoupled architecture (sec. 7.5): structure by variance, level by AVA.

The joint greedy of sec. 7.3 mixes an objective (the AVA gain, which
needs kappa, the Totem s and the day's book) with a constraint (the
variance cost, which needs only Sigma and the vegas). Sec. 7.5 separates
them into two runs:

Run 1 — the structure, portfolio-free. Hierarchical agglomerative
clustering of the test nodes on the *base-risk distance*

    d_ij = sqrt(Var(dsigma_i - dsigma_j))
         = sqrt(s_i^2 + s_j^2 - 2 rho_ij s_i s_j),

exactly the quantity of Theoreme 2 (i): fusing j onto pivot i costs
TE^2 = nu_j^2 d_ij^2. Merging by increasing base risk yields a
*dendrogram*; a netting scheme is a *cut* of that tree. Propriete 8
(portfolio-free criterion): if every node j of a set satisfies
d_{j,pivot} <= eps * s_j, then for ANY book
TE <= eps * sum |nu_j| s_j = eps * AVA_brut / kappa.

Run 2 — the level, by AVA. On the validated structure, production
evaluates AVA = kappa * sum_r |m_r| s~_r on the sets of the cut with the
day's Totem s — no combinatorial re-optimisation. The mandatory per-book
check is the single closed-form ratio TE^2 / Var(DeltaPi) (Def. 3); when
it fails, the cut is lowered (finer sets) for that book.

Sec. 7.4 — shock families and stability: the same partition is replayed
under alternative covariances (daily variations restricted to the
pillars, stressed correlations); fusions failing under any regime are
unstable and undone — realised here by lowering the cut until the scheme
passes under every covariance.

Merges are constrained to rectangular contiguous sets on the node grid
(interpretability, sec. 7.3). All variances use ordinary matrix products
and reductions (Property 1) — no Kronecker / tensor product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .datasource import MarketDataBundle
from .model import UncertaintyModel
from .netting import NettingScheme, SchemeEvaluation, SetStat
from .optimizer import Rect, _labels_from_rects, _mergeable, _union

__all__ = [
    "base_risk_distance",
    "ClusterMerge",
    "Dendrogram",
    "build_dendrogram",
    "cut_pivots",
    "evaluate_cut",
    "DecoupledResult",
    "decoupled_netting",
    "stability_report",
    "stable_cut",
]


# --------------------------------------------------------------------------- #
# Base-risk distance (Th. 2 (i) / sec. 7.5)
# --------------------------------------------------------------------------- #
def base_risk_distance(bundle: MarketDataBundle) -> np.ndarray:
    """Pairwise base risk d_ij between test nodes, as an (n, n) matrix on
    the row-major node indexing (i = a * qK + b).

    d_ij^2 = s_i^2 + s_j^2 - 2 rho_ij s_i s_j, with rho_ij the entrywise
    decoupled correlation rho_mat[a,a'] * rho_strike[b,b'] (annex) or the
    supplied full correlation. Entries are scalar products — this is a
    distance table, no Kronecker operator is built or used.
    """
    M, K, n = bundle.M, bundle.K, bundle.n
    s = bundle.s
    d2 = np.zeros((n, n))
    for i in range(n):
        a, b = divmod(i, K)
        for j in range(i + 1, n):
            ap, bp = divmod(j, K)
            if bundle.corr_full is not None:
                rho = bundle.corr_full[i, j]
            else:
                rho = bundle.corr_mat[a, ap] * bundle.corr_strike[b, bp]
            v = s[a, b] ** 2 + s[ap, bp] ** 2 - 2.0 * rho * s[a, b] * s[ap, bp]
            d2[i, j] = d2[j, i] = max(v, 0.0)
    return np.sqrt(d2)


def _set_epsilon(cells: list[tuple[int, int]], dist: np.ndarray, s: np.ndarray, K: int):
    """Best pivot of a set under the Propriete 8 criterion: the cell
    minimising max_j d_{j,pivot} / s_j. Returns (epsilon, pivot_cell)."""
    best_eps, best_pivot = np.inf, cells[0]
    for p in cells:
        ip = p[0] * K + p[1]
        eps = max(dist[c[0] * K + c[1], ip] / s[c] for c in cells)
        if eps < best_eps:
            best_eps, best_pivot = eps, p
    return best_eps, best_pivot


def _rect_cells(rect: Rect) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = rect
    return [(m, k) for m in range(r0, r1 + 1) for k in range(c0, c1 + 1)]


# --------------------------------------------------------------------------- #
# Run 1 — the dendrogram (portfolio-free)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterMerge:
    step: int
    rect_a: Rect
    rect_b: Rect
    merged: Rect
    height: float          # epsilon of the merged set (Prop. 8 criterion)


@dataclass(frozen=True)
class Dendrogram:
    """Full merge tree of Run 1. A netting scheme is a cut of this tree:
    apply the merges in order while height <= epsilon."""

    M: int
    K: int
    merges: list[ClusterMerge] = field(default_factory=list)

    def n_merges_at(self, epsilon: float) -> int:
        n = 0
        for mg in self.merges:
            if mg.height > epsilon:
                break
            n += 1
        return n

    def rects_after(self, n_merges: int) -> list[Rect]:
        rects: list[Rect] = [(m, m, k, k) for m in range(self.M) for k in range(self.K)]
        for mg in self.merges[:n_merges]:
            rects = [r for r in rects if r not in (mg.rect_a, mg.rect_b)]
            rects.append(mg.merged)
        return rects

    def cut(self, epsilon: float) -> np.ndarray:
        """(M, K) labels of the partition at cut height epsilon."""
        return self.labels_after(self.n_merges_at(epsilon))

    def labels_after(self, n_merges: int) -> np.ndarray:
        return _labels_from_rects(self.rects_after(n_merges), self.M, self.K)


def build_dendrogram(bundle: MarketDataBundle) -> Dendrogram:
    """Run 1 of sec. 7.5 — needs only Sigma (s, rho), never the book.

    Agglomerative clustering constrained to rectangular contiguous sets:
    at each step, merge the adjacent pair whose union has the smallest
    Propriete-8 epsilon (max_j d_{j,pivot}/s_j minimised over the pivot).
    """
    M, K = bundle.M, bundle.K
    dist = base_risk_distance(bundle)
    s = bundle.s
    rects: list[Rect] = [(m, m, k, k) for m in range(M) for k in range(K)]
    merges: list[ClusterMerge] = []
    while len(rects) > 1:
        best = None  # (eps, i, j, union)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if not _mergeable(rects[i], rects[j]):
                    continue
                union = _union(rects[i], rects[j])
                eps, _ = _set_epsilon(_rect_cells(union), dist, s, K)
                if best is None or eps < best[0]:
                    best = (eps, i, j, union)
        if best is None:
            break
        eps, i, j, union = best
        merges.append(
            ClusterMerge(
                step=len(merges) + 1,
                rect_a=rects[i],
                rect_b=rects[j],
                merged=union,
                height=eps,
            )
        )
        rects = [r for idx, r in enumerate(rects) if idx not in (i, j)] + [union]
    return Dendrogram(M=M, K=K, merges=merges)


def cut_pivots(bundle: MarketDataBundle, labels: np.ndarray) -> dict:
    """Structure pivots of a cut (portfolio-free, Prop. 8 criterion).

    Returns {set_id: {"pivot": (m, k), "epsilon": eps}} where eps is the
    realised max_j d_{j,pivot}/s_j of the set."""
    labels = np.asarray(labels, dtype=int)
    dist = base_risk_distance(bundle)
    out = {}
    for sid in range(int(labels.max()) + 1):
        cells = [tuple(c) for c in np.argwhere(labels == sid)]
        eps, pivot = _set_epsilon(cells, dist, bundle.s, bundle.K)
        out[sid] = {"pivot": pivot, "epsilon": float(eps)}
    return out


# --------------------------------------------------------------------------- #
# Run 2 — evaluate a cut with the day's book (level by AVA)
# --------------------------------------------------------------------------- #
def evaluate_cut(
    bundle: MarketDataBundle,
    labels: np.ndarray,
    alpha: float,
    kappa: float,
    pivots: Optional[dict] = None,
) -> SchemeEvaluation:
    """Run 2 of sec. 7.5: AVA and the variance test of a structure cut.

    The representative shock of each set is its *structure* pivot
    (portfolio-free, from Run 1) — not a book-dependent choice, so the
    scheme does not move with the book. Same matrix-form quantities as
    :meth:`NettingScheme.evaluate` (Property 1), s~_r = s at the pivot.
    """
    labels = np.asarray(labels, dtype=int)
    model = UncertaintyModel(bundle=bundle, kappa=kappa)
    if pivots is None:
        pivots = cut_pivots(bundle, labels)
    vega, s = bundle.vega, bundle.s
    n_sets = int(labels.max()) + 1

    proxy = np.zeros_like(vega, dtype=float)
    m = np.zeros(n_sets)
    s_tilde = np.zeros(n_sets)
    for sid in range(n_sets):
        mask = labels == sid
        m[sid] = float(vega[mask].sum())
        pm, pk = pivots[sid]["pivot"]
        s_tilde[sid] = float(s[pm, pk])
        proxy[pm, pk] += m[sid]
    residual = vega - proxy
    te2 = max(model.variance_of(residual), 0.0)
    var_total = model.var_total
    r2 = 1.0 - te2 / var_total if var_total > 0 else 1.0
    ava = float(kappa * np.sum(np.abs(m) * s_tilde))
    budget = model.budget(alpha)

    stats = []
    for sid in range(n_sets):
        mask = labels == sid
        stats.append(
            SetStat(
                set_id=sid,
                size=int(mask.sum()),
                net_vega=float(m[sid]),
                gross_vega=float(np.abs(vega[mask]).sum()),
                s_tilde=float(s_tilde[sid]),
                ava_netted=float(kappa * abs(m[sid]) * s_tilde[sid]),
                ava_addup=float(kappa * np.sum(np.abs(vega[mask]) * s[mask])),
            )
        )
    return SchemeEvaluation(
        scheme=NettingScheme.from_labels(labels, weighting="pivot"),
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
class DecoupledResult:
    """Output of the two-run architecture of sec. 7.5."""

    dendrogram: Dendrogram
    epsilon: float                 # requested cut height
    n_merges: int                  # merges actually applied
    labels: np.ndarray
    pivots: dict
    evaluation: SchemeEvaluation
    epsilon_realised: float        # max set epsilon of the cut
    te_bound: float                # Prop. 8: eps_realised * sum |nu| s
    lowered: int                   # merges undone by the per-book check

    @property
    def bound_is_tight(self) -> bool:
        return np.sqrt(max(self.evaluation.te2, 0.0)) <= self.te_bound + 1e-9


def decoupled_netting(
    bundle: MarketDataBundle,
    alpha: float,
    kappa: float,
    epsilon: float,
    dendrogram: Optional[Dendrogram] = None,
) -> DecoupledResult:
    """Sec. 7.5 end to end: cut the (portfolio-free) dendrogram at
    ``epsilon``, evaluate the day's book on the cut, and apply the
    mandatory per-book check — if the variance test (6) or the floor (7)
    fails, lower the cut (undo merges) until both pass."""
    dendro = dendrogram if dendrogram is not None else build_dendrogram(bundle)
    n = dendro.n_merges_at(epsilon)
    lowered = 0
    while True:
        labels = dendro.labels_after(n)
        pivots = cut_pivots(bundle, labels)
        ev = evaluate_cut(bundle, labels, alpha, kappa, pivots)
        if (ev.passes_variance and ev.passes_floor) or n == 0:
            break
        n -= 1
        lowered += 1
    eps_real = max((p["epsilon"] for p in pivots.values()), default=0.0)
    te_bound = float(eps_real * np.sum(np.abs(bundle.vega) * bundle.s))
    return DecoupledResult(
        dendrogram=dendro,
        epsilon=epsilon,
        n_merges=n,
        labels=labels,
        pivots=pivots,
        evaluation=ev,
        epsilon_realised=float(eps_real),
        te_bound=te_bound,
        lowered=lowered,
    )


# --------------------------------------------------------------------------- #
# Sec. 7.4 — stability across shock families
# --------------------------------------------------------------------------- #
def stability_report(
    labels: np.ndarray,
    base_bundle: MarketDataBundle,
    alternatives: dict[str, MarketDataBundle],
    alpha: float,
    kappa: float,
) -> list[dict]:
    """Replay the variance test of the SAME partition under alternative
    covariances (sec. 7.4: daily-variation windows restricted to the
    pillars via Propriete 5, stressed correlations, bid-ask...). Only the
    covariance changes — same transport matrices, same TE formulas.

    Pivots are the base structure's pivots: the representation under test
    is the production one."""
    pivots = cut_pivots(base_bundle, labels)
    rows = []
    for name, alt in {"base": base_bundle, **alternatives}.items():
        ev = evaluate_cut(alt, labels, alpha, kappa, pivots)
        rows.append(
            {
                "regime": name,
                "te2": ev.te2,
                "budget": ev.budget,
                "r2": ev.r2,
                "passes_variance": ev.passes_variance,
                "passes_floor": ev.passes_floor,
            }
        )
    return rows


def stable_cut(
    bundle: MarketDataBundle,
    alternatives: dict[str, MarketDataBundle],
    alpha: float,
    kappa: float,
    epsilon: float,
    dendrogram: Optional[Dendrogram] = None,
) -> dict:
    """Sec. 7.4 protocol on the dendrogram: keep the deepest cut at or
    below ``epsilon`` whose partition passes the variance test and the
    floor under the base covariance AND every alternative regime.
    Fusions failing any regime are unstable and undone (the regulatory
    asymmetry: refusing a valid fusion is allowed, keeping an invalidated
    one is not)."""
    dendro = dendrogram if dendrogram is not None else build_dendrogram(bundle)
    n = dendro.n_merges_at(epsilon)
    undone = 0
    while n >= 0:
        labels = dendro.labels_after(n)
        report = stability_report(labels, bundle, alternatives, alpha, kappa)
        # variance test must hold in every regime; the conservatism floor
        # is a prudential-level check, meaningful under the base covariance
        # only (sec. 7.4: Sigma_daily is not a prudential uncertainty level)
        ok = all(r["passes_variance"] for r in report) and report[0]["passes_floor"]
        if ok or n == 0:
            break
        n -= 1
        undone += 1
    return {
        "dendrogram": dendro,
        "n_merges": n,
        "undone": undone,
        "labels": labels,
        "report": report,
        "evaluation": evaluate_cut(bundle, labels, alpha, kappa),
    }
