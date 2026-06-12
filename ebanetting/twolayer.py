"""Two-layer optimal-netting methodology (companion note "Recherche du
netting optimal des sensibilités vega : méthodologie complète à deux
couches", June 2026).

The organising principle: the *disagreement* between vol points is a
property of the UNDERLYING (it knows nothing of the book), while the
RR/FLY aggregates are the projection of the BOOK on that structure.

Layer 1 — per underlying (periodic, no book):
  * deformation model (Def. 1):  dsigma_j = dsigma_niv + x_j dS
    + x_j^2 dC + eps_j, with common level / slope / curvature shocks
    (sd s0, sigma_S, sigma_C) and independent idiosyncratic noise
    (sd sigma_eps). Fitted by cross-sectional regression on daily data;
    R^2 measures its validity.
  * Theoreme 1 (pair base risk): d_ij^2 = (x_j-x_i)^2 sigma_S^2
    + (x_j-x_i)^2 (x_j+x_i)^2 sigma_C^2 + 2 sigma_eps^2 — it is the
    *gap* that matters, wings amplify curvature, the idio is a floor.
  * Propriete 2 (generated correlation): rho(x) = s0 / s_x — the
    correlation curve is fabricated by the model, not primitive.
  * dendrogram of contiguous strike groups by increasing disagreement,
    pivot-linkage, pivot = most liquid point: the candidate structure.

Layer 2 — per book (each AVA run):
  * exact group residual (Theoreme 2): with RR_p = sum nu_j (x_j - x_p)
    and FLY_p = sum nu_j (x_j^2 - x_p^2),
    Var(R) = RR_p^2 sigma_S^2 + FLY_p^2 sigma_C^2 + idio, where the
    idio block is sigma_eps^2 (sum_{j!=p} nu_j^2 + (sum_{j!=p} nu_j)^2)
    — signed sums for the common modes (compensation possible), squares
    for the idio (never compensates: the incompressible floor).
  * pivot translation (sec. 6.3): RR_p = RR_0 - m x_p,
    FLY_p = FLY_0 - m x_p^2; the vega barycenter x_p = RR_0 / m kills
    RR_p exactly — the slope term vanishes by choice of representation.
  * decisions N1-N4 (sec. 6.4): collapse if Var(R) fits, else extraction
    (keep m netted, carve RR/FLY out in add-up with s_skew / s_fly),
    else scission when the idio alone overflows.
  * the exact computation versus the layer-1 majorant
    sum |nu_j| d_jp (triangle inequality): the bar inside the sum erases
    the signs, the bar around keeps them — the exact test can authorise
    what the majorant wrongly refused, never the reverse.

Errors of groups within one tranche share dS and dC: coefficients are
summed across groups *before* squaring (the global TE is not the sum of
group variances). Everything is scalar products, ordinary matrix algebra
and reductions — no tensor products.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "SmileModel",
    "fit_smile_model",
    "model_distance",
    "generated_correlation",
    "TrancheMerge",
    "tranche_dendrogram",
    "cut_tranche",
    "group_projections",
    "barycenter_pivot",
    "group_residual_variance",
    "majorant_residual_sd",
    "book_variance",
    "GroupDecision",
    "TwoLayerResult",
    "evaluate_book",
]


# --------------------------------------------------------------------------- #
# Layer 1 — the underlying (no book)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SmileModel:
    """Deformation model of a tranche (Def. 1): level / slope / curvature
    common shocks plus independent idiosyncratic noise."""

    s0: float           # sd of the common level shock
    sigma_s: float      # sd of the slope (skew) shock dS
    sigma_c: float      # sd of the curvature shock dC
    sigma_eps: float    # sd of the per-point idiosyncratic noise
    r2: float = 1.0     # validity of the model on daily data

    def point_uncertainty(self, x) -> np.ndarray:
        """s_x = sqrt(s0^2 + x^2 sigma_S^2 + x^4 sigma_C^2 + sigma_eps^2)."""
        x = np.asarray(x, dtype=float)
        return np.sqrt(
            self.s0 ** 2
            + x ** 2 * self.sigma_s ** 2
            + x ** 4 * self.sigma_c ** 2
            + self.sigma_eps ** 2
        )


def fit_smile_model(x: Sequence[float], shocks: np.ndarray) -> SmileModel:
    """A1 — estimate the deformation model on a tranche history.

    Per date t, cross-sectional regression of the strike shocks on
    [1, x, x^2] (ordinary least squares): the coefficients are the level,
    slope and curvature shocks of that date; their sample sds give
    (s0, sigma_S, sigma_C), the residual variance gives sigma_eps^2 and
    the pooled R^2 the validity of the model."""
    x = np.asarray(x, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    design = np.column_stack([np.ones_like(x), x, x ** 2])      # (K, 3)
    coeffs, *_ = np.linalg.lstsq(design, shocks.T, rcond=None)  # (3, T)
    level, dS, dC = coeffs
    resid = shocks.T - design @ coeffs
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((shocks - shocks.mean()) ** 2))
    n_eff = max(resid.shape[1] * (resid.shape[0] - 3), 1)
    return SmileModel(
        s0=float(np.std(level)),
        sigma_s=float(np.std(dS)),
        sigma_c=float(np.std(dC)),
        sigma_eps=float(np.sqrt(ss_res / n_eff)),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
    )


def model_distance(x: Sequence[float], model: SmileModel) -> np.ndarray:
    """Theoreme 1 — the disagreement matrix D = (d_ij) generated by the
    model: d_ij^2 = (x_j-x_i)^2 sigma_S^2 + (x_j-x_i)^2 (x_j+x_i)^2
    sigma_C^2 + 2 sigma_eps^2 (zero on the diagonal).

    It is the gap |x_j - x_i| that matters, not the distance to the ATM;
    the (x_i + x_j) amplification shrinks the nettable neighbourhood in
    the wings; symmetric pairs lose the curvature term; the idio is a
    floor d^2 >= 2 sigma_eps^2."""
    x = np.asarray(x, dtype=float)
    K = x.size
    d2 = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            gap, amp = x[j] - x[i], x[j] + x[i]
            v = (
                gap ** 2 * model.sigma_s ** 2
                + gap ** 2 * amp ** 2 * model.sigma_c ** 2
                + 2.0 * model.sigma_eps ** 2
            )
            d2[i, j] = d2[j, i] = v
    return np.sqrt(d2)


def generated_correlation(x: Sequence[float], model: SmileModel) -> np.ndarray:
    """Propriete 2 — the correlation of each point with the ATM is not a
    primitive input, the model fabricates it: rho(x) = s0 / s_x."""
    return model.s0 / model.point_uncertainty(x)


@dataclass(frozen=True)
class TrancheMerge:
    step: int
    members_a: tuple        # strike indices of the two merged groups
    members_b: tuple
    merged: tuple
    pivot: int              # most liquid point of the merged group
    height: float           # pivot-to-pivot disagreement d


def tranche_dendrogram(
    x: Sequence[float],
    dist: np.ndarray,
    s: Sequence[float],
) -> list[TrancheMerge]:
    """A3 — hierarchical clustering of contiguous strike groups by
    increasing disagreement (pivot linkage: d between group pivots),
    the pivot of a group being its most liquid point (smallest s).
    The output tree is independent of the book."""
    s = np.asarray(s, dtype=float)
    groups: list[tuple] = [(i,) for i in range(len(x))]
    pivots: list[int] = list(range(len(x)))
    merges: list[TrancheMerge] = []
    while len(groups) > 1:
        best = None  # (height, position)
        for g in range(len(groups) - 1):
            h = float(dist[pivots[g], pivots[g + 1]])
            if best is None or h < best[0]:
                best = (h, g)
        h, g = best
        merged = groups[g] + groups[g + 1]
        pivot = merged[int(np.argmin(s[list(merged)]))]
        merges.append(
            TrancheMerge(
                step=len(merges) + 1,
                members_a=groups[g],
                members_b=groups[g + 1],
                merged=merged,
                pivot=pivot,
                height=h,
            )
        )
        groups[g : g + 2] = [merged]
        pivots[g : g + 2] = [pivot]
    return merges


def cut_tranche(merges: list[TrancheMerge], n_points: int, epsilon: float) -> list[tuple]:
    """Cut of the tranche dendrogram: apply the merges in order while
    height <= epsilon. Returns the groups (tuples of strike indices)."""
    groups = [(i,) for i in range(n_points)]
    for mg in merges:
        if mg.height > epsilon:
            break
        groups = [g for g in groups if g not in (mg.members_a, mg.members_b)]
        groups.append(mg.merged)
    return sorted(groups)


# --------------------------------------------------------------------------- #
# Layer 2 — the book
# --------------------------------------------------------------------------- #
def group_projections(nu: Sequence[float], x: Sequence[float], x_pivot: float) -> dict:
    """N1 — three scalar products: net level m, net risk-reversal RR_p
    and net butterfly FLY_p relative to the pivot coordinate."""
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    return {
        "m": float(nu.sum()),
        "rr": float(nu @ (x - x_pivot)),
        "fly": float(nu @ (x ** 2 - x_pivot ** 2)),
    }


def barycenter_pivot(nu: Sequence[float], x: Sequence[float]) -> Optional[float]:
    """Sec. 6.3 — the optimal pivot: x_p = RR_0 / m, the vega-weighted
    barycenter of the strikes, kills RR_p exactly (RR_p = RR_0 - m x_p).
    None when the net level vanishes (no finite barycenter)."""
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    m = float(nu.sum())
    if abs(m) < 1e-300:
        return None
    return float(nu @ x / m)


def group_residual_variance(
    nu: Sequence[float],
    x: Sequence[float],
    pivot_index: int,
    model: SmileModel,
) -> float:
    """Theoreme 2 — exact variance of the group residual, no inequality:

    Var(R) = RR_p^2 sigma_S^2 + FLY_p^2 sigma_C^2
             + sigma_eps^2 (sum_{j!=p} nu_j^2 + (sum_{j!=p} nu_j)^2),

    the last term being the shared idio of the pivot. Common modes enter
    through signed sums (compensation possible); the idio through squares
    (never compensates)."""
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    proj = group_projections(nu, x, x[pivot_index])
    others = np.delete(nu, pivot_index)
    idio = model.sigma_eps ** 2 * (float(others @ others) + float(others.sum()) ** 2)
    return (
        proj["rr"] ** 2 * model.sigma_s ** 2
        + proj["fly"] ** 2 * model.sigma_c ** 2
        + idio
    )


def majorant_residual_sd(
    nu: Sequence[float],
    x: Sequence[float],
    pivot_index: int,
    model: SmileModel,
) -> float:
    """The layer-1 majorant sum_{j!=p} |nu_j| d_{j,pivot} (triangle
    inequality): unconditionally valid, never tight — it erases the
    signs, so it double-counts what actually cancels."""
    nu = np.asarray(nu, dtype=float)
    dist = model_distance(x, model)
    return float(
        sum(abs(nu[j]) * dist[j, pivot_index] for j in range(len(nu)) if j != pivot_index)
    )


def book_variance(nu: Sequence[float], x: Sequence[float], model: SmileModel) -> float:
    """Var(DeltaPi) of the tranche under the model: the common level,
    slope and curvature enter through the signed aggregates, the idio
    through the squares —
    m^2 s0^2 + RR_0^2 sigma_S^2 + FLY_0^2 sigma_C^2 + sigma_eps^2 sum nu^2."""
    nu = np.asarray(nu, dtype=float)
    proj = group_projections(nu, x, 0.0)
    return (
        proj["m"] ** 2 * model.s0 ** 2
        + proj["rr"] ** 2 * model.sigma_s ** 2
        + proj["fly"] ** 2 * model.sigma_c ** 2
        + model.sigma_eps ** 2 * float(nu @ nu)
    )


@dataclass(frozen=True)
class GroupDecision:
    members: tuple
    pivot_index: int
    m: float
    rr: float
    fly: float
    var_exact: float          # Th. 2 standalone residual variance
    majorant_sd: float        # layer-1 bound on the residual sd
    decision: str             # "collapse" | "extract" | "split"
    ava: float                # contribution to the AVA
    s_pivot: float


@dataclass(frozen=True)
class TwoLayerResult:
    decisions: list[GroupDecision]
    te2: float                # exact global TE^2 (coefficients summed)
    te2_majorant: float       # (sum |nu_j| d_jp)^2 — what layer 1 alone sees
    budget: float
    var_total: float          # Var(DeltaPi) under the model
    passes_variance: bool
    ava: float
    ava_brut: float           # kappa sum |nu_j| s_xj
    ava_floor: float          # kappa sqrt(Var(DeltaPi))
    passes_floor: bool
    alpha: float
    kappa: float

    @property
    def admissible(self) -> bool:
        return self.passes_variance and self.passes_floor

    @property
    def ava_saving_pct(self) -> float:
        return 1.0 - self.ava / self.ava_brut if self.ava_brut > 0 else 0.0


def _global_te2(decisions: list[GroupDecision], model: SmileModel,
                nu: np.ndarray, x: np.ndarray) -> float:
    """Exact global TE^2: groups of one tranche share dS and dC, so the
    RR / FLY coefficients are summed across collapsed groups *before*
    squaring; extracted groups provision RR/FLY separately and leave
    only their idio in the residual."""
    rr_tot = sum(d.rr for d in decisions if d.decision == "collapse")
    fly_tot = sum(d.fly for d in decisions if d.decision == "collapse")
    idio = 0.0
    for d in decisions:
        others = np.delete(nu[list(d.members)],
                           list(d.members).index(d.pivot_index))
        idio += model.sigma_eps ** 2 * (float(others @ others) + float(others.sum()) ** 2)
    return rr_tot ** 2 * model.sigma_s ** 2 + fly_tot ** 2 * model.sigma_c ** 2 + idio


def _decide(members: tuple, nu: np.ndarray, x: np.ndarray, model: SmileModel,
            kappa: float, s_skew: float, s_fly: float,
            decision: str = "collapse") -> GroupDecision:
    nu_g, x_g = nu[list(members)], x[list(members)]
    bary = barycenter_pivot(nu_g, x_g)
    target = bary if bary is not None else float(x_g.mean())
    local_pivot = int(np.argmin(np.abs(x_g - target)))   # nearest quoted node
    pivot_index = members[local_pivot]
    proj = group_projections(nu_g, x_g, x[pivot_index])
    s_pivot = float(model.point_uncertainty(x[pivot_index]))
    ava = kappa * abs(proj["m"]) * s_pivot
    if decision == "extract":
        ava += kappa * (abs(proj["rr"]) * s_skew + abs(proj["fly"]) * s_fly)
    return GroupDecision(
        members=members,
        pivot_index=pivot_index,
        m=proj["m"],
        rr=proj["rr"],
        fly=proj["fly"],
        var_exact=group_residual_variance(nu_g, x_g, local_pivot, model),
        majorant_sd=majorant_residual_sd(nu_g, x_g, local_pivot, model),
        decision=decision,
        ava=ava,
        s_pivot=s_pivot,
    )


def evaluate_book(
    groups: list[tuple],
    nu: Sequence[float],
    x: Sequence[float],
    model: SmileModel,
    alpha: float,
    kappa: float,
    s_skew: Optional[float] = None,
    s_fly: Optional[float] = None,
) -> TwoLayerResult:
    """B1-B3 — layer 2 on the candidate structure of layer 1.

    Never all-or-nothing (N2-N4): all groups start collapsed on their
    barycenter pivot; while the *global* exact test fails, the group with
    the largest standalone residual variance is moved to extraction
    (its m stays netted, RR/FLY are carved out in add-up with s_skew /
    s_fly — by default the slope / curvature shock sds, so the carve-out
    provisions exactly the extracted risk); if the idio alone still
    overflows, the worst remaining group is split in two (scission).
    Then the global test (6), the floor (7) and the AVA."""
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    s_skew = model.sigma_s if s_skew is None else s_skew
    s_fly = model.sigma_c if s_fly is None else s_fly
    var_total = book_variance(nu, x, model)
    budget = (1.0 - alpha) * var_total

    decisions = [_decide(tuple(g), nu, x, model, kappa, s_skew, s_fly) for g in groups]

    def te2() -> float:
        return _global_te2(decisions, model, nu, x)

    # N3 — extraction, worst standalone variance first
    while te2() > budget:
        candidates = [i for i, d in enumerate(decisions)
                      if d.decision == "collapse" and (d.rr != 0.0 or d.fly != 0.0)]
        if not candidates:
            break
        worst = max(candidates, key=lambda i: decisions[i].var_exact)
        decisions[worst] = _decide(decisions[worst].members, nu, x, model,
                                   kappa, s_skew, s_fly, decision="extract")
    # N4 — scission when the idio alone overflows (illiquid wings)
    guard = 0
    while te2() > budget and guard < 2 * len(nu):
        guard += 1
        splittable = [i for i, d in enumerate(decisions) if len(d.members) > 1]
        if not splittable:
            break
        worst = max(splittable, key=lambda i: decisions[i].var_exact)
        members = decisions[worst].members
        half = len(members) // 2
        # each piece takes its own (local barycenter) pivot — Th. 1:
        # small gap means small d wherever the piece sits
        decisions[worst : worst + 1] = [
            _decide(tuple(p), nu, x, model, kappa, s_skew, s_fly)
            for p in (members[:half], members[half:])
        ]

    final_te2 = te2()
    ava = float(sum(d.ava for d in decisions))
    ava_brut = float(kappa * np.sum(np.abs(nu) * model.point_uncertainty(x)))
    ava_floor = float(kappa * np.sqrt(max(var_total, 0.0)))
    majorant = sum(d.majorant_sd for d in decisions if d.decision == "collapse")
    return TwoLayerResult(
        decisions=decisions,
        te2=final_te2,
        te2_majorant=float(majorant ** 2),
        budget=budget,
        var_total=var_total,
        passes_variance=final_te2 <= budget + 1e-12,
        ava=ava,
        ava_brut=ava_brut,
        ava_floor=ava_floor,
        passes_floor=ava >= ava_floor - 1e-9,
        alpha=alpha,
        kappa=kappa,
    )
