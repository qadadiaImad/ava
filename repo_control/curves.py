"""Zero curves, discrete dividends, forwards and the exact implied-repo bootstrap.

Conventions
-----------
* Times are year fractions from the valuation date.
* Rates and repo are continuously compounded zero rates.  Curves interpolate
  linearly in the cumulative integral ``I(t) = z(t) * t`` (flat forwards between
  pillars), which is also the natural "carry" space used by the detector.
* The forward with discrete cash dividends ``d_i`` paid at ex-dates ``t_i`` is

      F(T) = G(T) * (S - sum_{t_i <= T} d_i / G(t_i)),   G(t) = exp(R(t) - B(t))

  with ``R(t) = r(t) t`` and ``B(t) = b(t) t`` the cumulative rate and repo.
  ``D(T) = sum_{t_i <= T} d_i / G(t_i)`` is called the carry-discounted dividend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve with flat forwards between pillars."""

    times: np.ndarray
    zeros: np.ndarray

    def __post_init__(self) -> None:
        t = np.atleast_1d(np.asarray(self.times, dtype=float))
        z = np.atleast_1d(np.asarray(self.zeros, dtype=float))
        if t.shape != z.shape or t.ndim != 1 or t.size == 0:
            raise ValueError("times and zeros must be 1-d arrays of equal length")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing and positive")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "zeros", z)

    @classmethod
    def flat(cls, rate: float, horizon: float = 50.0) -> "ZeroCurve":
        return cls(np.array([horizon]), np.array([rate]))

    @classmethod
    def from_cumulative(cls, times: np.ndarray, cumulative: np.ndarray) -> "ZeroCurve":
        times = np.asarray(times, dtype=float)
        return cls(times, np.asarray(cumulative, dtype=float) / times)

    @property
    def cumulative(self) -> np.ndarray:
        """Cumulative integral ``z_k * T_k`` at the pillars."""
        return self.zeros * self.times

    def cum(self, t) -> np.ndarray:
        """Cumulative integral ``I(t)``; the last forward is extended beyond the last pillar."""
        t = np.asarray(t, dtype=float)
        knots_t = np.concatenate(([0.0], self.times))
        knots_i = np.concatenate(([0.0], self.cumulative))
        out = np.interp(t, knots_t, knots_i)
        last_fwd = (knots_i[-1] - knots_i[-2]) / (knots_t[-1] - knots_t[-2])
        return np.where(t > knots_t[-1], knots_i[-1] + last_fwd * (t - knots_t[-1]), out)

    def zero(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.cum(t) / t

    def df(self, t) -> np.ndarray:
        return np.exp(-self.cum(t))

    def bumped(self, index: int, size: float) -> "ZeroCurve":
        zeros = self.zeros.copy()
        zeros[index] += size
        return ZeroCurve(self.times, zeros)

    def shifted(self, size: float) -> "ZeroCurve":
        return ZeroCurve(self.times, self.zeros + size)


@dataclass(frozen=True)
class Dividends:
    """Discrete cash dividends, sorted by ex-date."""

    ex_times: np.ndarray
    amounts: np.ndarray

    def __post_init__(self) -> None:
        t = np.atleast_1d(np.asarray(self.ex_times, dtype=float))
        a = np.atleast_1d(np.asarray(self.amounts, dtype=float))
        if t.shape != a.shape:
            raise ValueError("ex_times and amounts must have equal length")
        order = np.argsort(t, kind="stable")
        object.__setattr__(self, "ex_times", t[order])
        object.__setattr__(self, "amounts", a[order])

    @classmethod
    def empty(cls) -> "Dividends":
        return cls(np.zeros(0), np.zeros(0))

    def __len__(self) -> int:
        return self.ex_times.size

    def window(self, start: float, end: float) -> np.ndarray:
        """Boolean mask of dividends with ``start < t_i <= end``."""
        return (self.ex_times > start) & (self.ex_times <= end)

    def shifted(self, dt: float) -> "Dividends":
        return Dividends(self.ex_times + dt, self.amounts)

    def scaled(self, factor) -> "Dividends":
        return Dividends(self.ex_times, self.amounts * factor)


def carry_dividends(spot: float, rates: ZeroCurve, repo: ZeroCurve, divs: Dividends, T) -> np.ndarray:
    """Carry-discounted cumulative dividends ``D(T) = sum_{t_i <= T} d_i / G(t_i)``."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if len(divs) == 0:
        return np.zeros_like(T)
    log_growth_i = rates.cum(divs.ex_times) - repo.cum(divs.ex_times)
    pv = divs.amounts * np.exp(-log_growth_i)
    mask = divs.ex_times[None, :] <= T[:, None]
    return mask @ pv


def forward(spot: float, rates: ZeroCurve, repo: ZeroCurve, divs: Dividends, T) -> np.ndarray:
    """Equity forward with discrete cash dividends and a repo (borrow) curve."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    log_growth = rates.cum(T) - repo.cum(T)
    return np.exp(log_growth) * (spot - carry_dividends(spot, rates, repo, divs, T))


def bootstrap_repo(
    spot: float,
    rates: ZeroCurve,
    divs: Dividends,
    pillars,
    forwards,
    tol: float = 1e-13,
    max_iter: int = 60,
) -> ZeroCurve:
    """Exact implied repo curve from forwards, rates and dividends.

    Pillars are solved sequentially.  With flat forward repo between pillars the
    model forward at pillar ``k`` is strictly decreasing in the unknown cumulative
    repo ``B_k`` (``dF/dB_k <= -F < 0``), so a safeguarded Newton step on
    ``ln F`` converges in a handful of iterations.
    """
    T = np.asarray(pillars, dtype=float)
    F = np.asarray(forwards, dtype=float)
    if T.shape != F.shape:
        raise ValueError("pillars and forwards must have equal length")
    R = rates.cum(T)
    R_div = rates.cum(divs.ex_times) if len(divs) else np.zeros(0)
    B = np.zeros_like(T)
    t_prev, b_prev = 0.0, 0.0
    for k in range(T.size):
        t_k = T[k]
        early = divs.ex_times <= t_prev
        mid = divs.window(t_prev, t_k)
        if np.any(early):
            knots_t = np.concatenate(([0.0], T[:k]))
            knots_b = np.concatenate(([0.0], B[:k]))
            b_early = np.interp(divs.ex_times[early], knots_t, knots_b)
            # d_i * exp((R_k - R_i) + B_i); the -B_k factor is applied inside the loop
            early_terms = divs.amounts[early] * np.exp(R[k] - R_div[early] + b_early)
        else:
            early_terms = np.zeros(0)
        w = (divs.ex_times[mid] - t_prev) / (t_k - t_prev)
        mid_amounts = divs.amounts[mid]
        mid_base = R[k] - R_div[mid]

        def model(x: float) -> tuple[float, float]:
            e_spot = spot * np.exp(R[k] - x)
            e_early = early_terms * np.exp(-x)
            e_mid = mid_amounts * np.exp(mid_base - (1.0 - w) * (x - b_prev))
            value = e_spot - e_early.sum() - e_mid.sum()
            deriv = -e_spot + e_early.sum() + ((1.0 - w) * e_mid).sum()
            return value, deriv

        # initial guess: treat in-period dividends with the previous repo level
        approx_div = early_terms.sum() * np.exp(-R[k]) + (
            mid_amounts * np.exp(-R_div[mid] + b_prev)
        ).sum()
        x = R[k] - np.log(F[k] / max(spot - approx_div, 1e-12 * spot))
        lo, hi = -np.inf, np.inf
        for _ in range(max_iter):
            value, deriv = model(x)
            if value <= 0:
                # too much repo: forward would be non-positive -> move left
                hi = min(hi, x)
                x = x - 0.5 if not np.isfinite(lo) else 0.5 * (lo + hi)
                continue
            g = np.log(value) - np.log(F[k])
            if g > 0:
                lo = max(lo, x)
            else:
                hi = min(hi, x)
            step = g / (deriv / value)
            x_new = x - step
            if not (lo < x_new < hi) and np.isfinite(lo) and np.isfinite(hi):
                x_new = 0.5 * (lo + hi)
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        B[k] = x
        t_prev, b_prev = t_k, x
    return ZeroCurve.from_cumulative(T, B)


def implied_repo_simple(spot: float, rate: float, forward_: float, div_pv: float, T: float) -> float:
    """Single-pillar textbook inversion ``b = r - ln(F / (S - D)) / T`` (D discounted at r)."""
    return rate - np.log(forward_ / (spot - div_pv)) / T
