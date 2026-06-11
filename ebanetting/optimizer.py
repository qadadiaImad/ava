"""Optimal netting under the variance budget (sec. 7 of the note).

The exact problem (10) is NP-hard (Bell-number combinatorics), so the
search space is restricted to *rectangular contiguous pavings* of the
node grid (sec. 7.3 — interpretable, documentable under art. 9(5)) and
solved with the greedy agglomerative algorithm of sec. 7.3:

    start from singletons, repeatedly merge the pair of adjacent
    rectangles maximising  (AVA reduction) / (TE^2 cost),  zero-cost
    merges first, while TE^2 stays within B = (1 - alpha) Var(DeltaPi);
    then verify the conservatism floor (7), rolling back if needed.

Also provides the Lagrangian frontier (complementary diagnostic) and the
adverse correlation stress / robust-fusion procedure (step 4, sec. 7.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .datasource import MarketDataBundle, nearest_correlation
from .model import UncertaintyModel
from .netting import NettingScheme, SchemeEvaluation, Weighting

__all__ = [
    "Rect",
    "GreedyResult",
    "MergeStep",
    "greedy_netting",
    "lagrangian_frontier",
    "stress_bundle",
    "robust_netting",
]

Rect = tuple[int, int, int, int]  # (r0, r1, c0, c1) inclusive


def _labels_from_rects(rects: list[Rect], M: int, K: int) -> np.ndarray:
    labels = np.full((M, K), -1, dtype=int)
    for i, (r0, r1, c0, c1) in enumerate(rects):
        labels[r0 : r1 + 1, c0 : c1 + 1] = i
    assert (labels >= 0).all(), "rectangles do not pave the grid"
    return labels


def _mergeable(a: Rect, b: Rect) -> bool:
    """True when the union of two rectangles is a rectangle."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    same_rows = ar0 == br0 and ar1 == br1
    same_cols = ac0 == bc0 and ac1 == bc1
    cols_adjacent = ac1 + 1 == bc0 or bc1 + 1 == ac0
    rows_adjacent = ar1 + 1 == br0 or br1 + 1 == ar0
    return (same_rows and cols_adjacent) or (same_cols and rows_adjacent)


def _union(a: Rect, b: Rect) -> Rect:
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


@dataclass(frozen=True)
class MergeStep:
    step: int
    rect_a: Rect
    rect_b: Rect
    gain: float           # AVA reduction
    cost: float           # TE^2 increase
    te2: float            # cumulative TE^2 after the merge
    ava: float
    n_sets: int


@dataclass(frozen=True)
class GreedyResult:
    scheme: NettingScheme
    evaluation: SchemeEvaluation
    history: list[MergeStep] = field(default_factory=list)
    rolled_back: int = 0


def _greedy(
    model: UncertaintyModel,
    weighting: Weighting,
    alpha: float,
    budget: Optional[float],
    mu: float = 0.0,
) -> GreedyResult:
    """Shared engine. With ``budget`` set, runs the constrained algorithm of
    sec. 7.3; with ``budget=None`` and ``mu > 0``, runs the Lagrangian
    relaxation min AVA + mu TE^2 (complementary), merging while gain - mu*cost > 0.
    """
    M, K = model.bundle.M, model.bundle.K
    rects: list[Rect] = [(m, m, k, k) for m in range(M) for k in range(K)]
    current = NettingScheme.from_labels(_labels_from_rects(rects, M, K), weighting)
    ev = current.evaluate(model, alpha)
    history: list[MergeStep] = []
    eps = 1e-12 * max(model.var_total, 1.0)

    while len(rects) > 1:
        best = None  # (priority, score, i, j, gain, cost, ev_new)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if not _mergeable(rects[i], rects[j]):
                    continue
                trial = rects[: i] + rects[i + 1 : j] + rects[j + 1 :] + [_union(rects[i], rects[j])]
                scheme = NettingScheme.from_labels(_labels_from_rects(trial, M, K), weighting)
                ev_new = scheme.evaluate(model, alpha)
                gain = ev.ava - ev_new.ava
                cost = ev_new.te2 - ev.te2
                if budget is not None:
                    if ev_new.te2 > budget + 1e-12:
                        continue
                    if cost <= eps and gain >= 0:
                        cand = (1, gain, i, j, gain, cost, ev_new)
                    elif gain > 0 and cost > eps:
                        cand = (0, gain / cost, i, j, gain, cost, ev_new)
                    else:
                        continue
                else:  # Lagrangian mode
                    score = gain - mu * cost
                    if score <= 0:
                        continue
                    cand = (0, score, i, j, gain, cost, ev_new)
                if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand
        if best is None:
            break
        _, _, i, j, gain, cost, ev_new = best
        merged = _union(rects[i], rects[j])
        ra, rb = rects[i], rects[j]
        rects = [r for idx, r in enumerate(rects) if idx not in (i, j)] + [merged]
        ev = ev_new
        history.append(
            MergeStep(
                step=len(history) + 1,
                rect_a=ra,
                rect_b=rb,
                gain=gain,
                cost=cost,
                te2=ev.te2,
                ava=ev.ava,
                n_sets=len(rects),
            )
        )

    scheme = NettingScheme.from_labels(_labels_from_rects(rects, M, K), weighting)
    return GreedyResult(scheme=scheme, evaluation=scheme.evaluate(model, alpha), history=history)


def greedy_netting(
    model: UncertaintyModel,
    alpha: float,
    weighting: Weighting = "pivot",
) -> GreedyResult:
    """Constrained greedy of sec. 7.3, including the floor roll-back (step 3)."""
    result = _greedy(model, weighting, alpha, budget=model.budget(alpha))
    # Step 3: conservatism floor (7). Roll merges back from the end until met.
    rolled = 0
    history = list(result.history)
    while not result.evaluation.passes_floor and history:
        history.pop()
        rolled += 1
        result = _replay(model, weighting, alpha, history)
    if rolled:
        result = replace(result, history=history, rolled_back=rolled)
    return result


def _replay(
    model: UncertaintyModel, weighting: Weighting, alpha: float, history: list[MergeStep]
) -> GreedyResult:
    M, K = model.bundle.M, model.bundle.K
    rects: list[Rect] = [(m, m, k, k) for m in range(M) for k in range(K)]
    for step in history:
        rects = [r for r in rects if r not in (step.rect_a, step.rect_b)]
        rects.append(_union(step.rect_a, step.rect_b))
    scheme = NettingScheme.from_labels(_labels_from_rects(rects, M, K), weighting)
    return GreedyResult(
        scheme=scheme, evaluation=scheme.evaluate(model, alpha), history=history
    )


def lagrangian_frontier(
    model: UncertaintyModel,
    alpha: float,
    mus: np.ndarray,
    weighting: Weighting = "pivot",
) -> list[dict]:
    """Sweep the shadow price mu of residual variance and trace the
    AVA / fidelity efficient frontier (complementary diagnostic)."""
    points = []
    for mu in mus:
        res = _greedy(model, weighting, alpha, budget=None, mu=float(mu))
        points.append(
            {
                "mu": float(mu),
                "ava": res.evaluation.ava,
                "te2": res.evaluation.te2,
                "r2": res.evaluation.r2,
                "n_sets": res.scheme.n_sets,
            }
        )
    return points


def stress_bundle(bundle: MarketDataBundle, delta: float) -> MarketDataBundle:
    """Adverse correlation stress rho -> max(rho - delta, -1) on off-diagonal
    terms (algorithm step 4, sec. 7.3): the stress must *reduce* intra-set
    correlation, i.e. work against the netting."""

    def stress(rho: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if rho is None:
            return None
        out = np.clip(rho - delta, -1.0, 1.0)
        np.fill_diagonal(out, 1.0)
        return nearest_correlation(out)

    return MarketDataBundle(
        tenors=bundle.tenors,
        tenor_years=bundle.tenor_years,
        strikes=bundle.strikes,
        vega=bundle.vega,
        s=bundle.s,
        corr_mat=stress(bundle.corr_mat),
        corr_strike=stress(bundle.corr_strike),
        corr_full=stress(bundle.corr_full),
        meta={**bundle.meta, "stress_delta": delta},
    )


def robust_netting(
    bundle: MarketDataBundle,
    alpha: float,
    kappa: float,
    deltas: list[float],
    weighting: Weighting = "pivot",
) -> dict:
    """Step 4 of the algorithm: re-run the greedy under correlation stress
    and keep only the fusions that survive every regime — the retained
    partition is the common refinement of the per-regime partitions,
    re-evaluated under the base model."""
    base_model = UncertaintyModel(bundle=bundle, kappa=kappa)
    runs = {0.0: greedy_netting(base_model, alpha, weighting)}
    for d in deltas:
        stressed = UncertaintyModel(bundle=stress_bundle(bundle, d), kappa=kappa)
        runs[d] = greedy_netting(stressed, alpha, weighting)

    # Common refinement: buckets stay together only if together in all runs.
    stacked = np.stack([r.scheme.labels for r in runs.values()], axis=-1)
    flat = stacked.reshape(-1, stacked.shape[-1])
    _, combined = np.unique(flat, axis=0, return_inverse=True)
    robust_labels = combined.reshape(bundle.M, bundle.K)
    robust_scheme = NettingScheme.from_labels(robust_labels, weighting)
    return {
        "runs": runs,
        "robust_scheme": robust_scheme,
        "robust_evaluation": robust_scheme.evaluate(base_model, alpha),
    }
