"""Div-agnostic repo control from forward and dividend controls.

For any exposure that depends on the underlying only through its forwards
(Europeans, forwards, futures), first order gives

    sens_D . dD + sens_b . db  =  dV/dF . dF

so the *total* of the dividend and repo lines does not depend on which
dividend panel was used; only the split does.  Exactly:

    forward line  = sens_b . (b(F_totem, D_FO) - b_FO)      # Totem forwards, FO dividends
    repo line     = sens_b . (b(F_totem, D_totem) - b_FO)
    div line      = sens_D . dLevel
    div + repo    = forward line + g . dLevel   (first order)

with ``J[k, y] = d b_k / d Level_y`` at fixed forwards (from the bootstrap) and
the non-forward dividend exposure

    g = sens_D - sens_D_pred,   sens_D_pred = -sens_b . J.

A forward-only book has ``g = 0``: its total control is the forward line and
needs no Totem dividends.  ``g`` comes from sensitivities alone (dividend
swaps, early exercise, repo-only legs, discrete-dividend exotics) and sizes
how much a bad repo/dividend split can cost a given book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .buckets import BucketSchedule, allocate
from .curves import Dividends, ZeroCurve, bootstrap_repo


def implied_repo_jacobian(
    spot: float,
    rates: ZeroCurve,
    schedule: BucketSchedule,
    levels,
    profile: Dividends,
    pillars,
    forwards,
    h: float = 1e-4,
) -> np.ndarray:
    """``J[k, y] = d b_k / d Level_y`` holding the forwards fixed (central differences on the bootstrap)."""
    levels = np.asarray(levels, dtype=float)
    J = np.zeros((np.size(pillars), levels.size))
    for y in range(levels.size):
        up, dn = levels.copy(), levels.copy()
        up[y] += h
        dn[y] -= h
        b_up = bootstrap_repo(spot, rates, allocate(schedule, up, profile), pillars, forwards).zeros
        b_dn = bootstrap_repo(spot, rates, allocate(schedule, dn, profile), pillars, forwards).zeros
        J[:, y] = (b_up - b_dn) / (2 * h)
    return J


def approx_repo_jacobian(pillars, alloc, spot: float, carry_div) -> np.ndarray:
    """Leading-order Jacobian ``-A[k, y] / (T_k (S - D_k))`` (dividends discounted at the rate only)."""
    T = np.asarray(pillars, dtype=float)
    return -np.asarray(alloc, dtype=float) / (T * (spot - np.asarray(carry_div, dtype=float)))[:, None]


def predicted_div_sensitivity(sens_repo, repo_jacobian) -> np.ndarray:
    """Dividend sensitivity a forward-only book would have, implied by its repo sensitivity."""
    return -np.asarray(sens_repo, dtype=float) @ np.asarray(repo_jacobian, dtype=float)


@dataclass(frozen=True)
class ControlLines:
    div_line: float
    repo_line: float
    forward_line: float
    non_forward_div_line: float

    @property
    def total(self) -> float:
        return self.div_line + self.repo_line

    @property
    def offset_ratio(self) -> float:
        """Close to 1 when the dividend and repo lines mostly cancel each other (a split artefact)."""
        gross = abs(self.div_line) + abs(self.repo_line)
        return 0.0 if gross == 0 else 1.0 - abs(self.total) / gross


def control_lines(sens_repo, sens_div, d_repo, d_repo_fwd, d_levels, repo_jacobian) -> ControlLines:
    """First-order control lines; every delta is Totem minus FO.

    ``d_repo``: implied repo from Totem forwards and Totem dividends, minus FO repo.
    ``d_repo_fwd``: implied repo from Totem forwards and *FO* dividends, minus FO repo.
    """
    sens_repo = np.asarray(sens_repo, dtype=float)
    sens_div = np.asarray(sens_div, dtype=float)
    d_levels = np.asarray(d_levels, dtype=float)
    g = sens_div - predicted_div_sensitivity(sens_repo, repo_jacobian)
    return ControlLines(
        div_line=float(sens_div @ d_levels),
        repo_line=float(sens_repo @ np.asarray(d_repo, dtype=float)),
        forward_line=float(sens_repo @ np.asarray(d_repo_fwd, dtype=float)),
        non_forward_div_line=float(g @ d_levels),
    )


@dataclass(frozen=True)
class SplitMateriality:
    non_forward_sens: np.ndarray
    split_sd: float
    div_line_sd: float

    @property
    def cancellation(self) -> float:
        """Share of the dividend-line uncertainty that cancels against the repo line."""
        return 0.0 if self.div_line_sd == 0 else 1.0 - self.split_sd / self.div_line_sd


def split_materiality(sens_div, sens_repo, repo_jacobian, level_cov) -> SplitMateriality:
    """1-sd P&L impact of dividend-consensus uncertainty on the total versus on the div line alone."""
    sens_div = np.asarray(sens_div, dtype=float)
    g = sens_div - predicted_div_sensitivity(sens_repo, repo_jacobian)
    C = np.asarray(level_cov, dtype=float)
    return SplitMateriality(g, float(np.sqrt(max(g @ C @ g, 0.0))), float(np.sqrt(max(sens_div @ C @ sens_div, 0.0))))
